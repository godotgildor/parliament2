#!/usr/bin/env python

import argparse
import subprocess
import logging
import multiprocessing
import os
import time
import tempfile
import shutil
import re


SV_CALLER_WEIGHTS = {
    "breakdancer": 1.0,
    "cnvnator": 1.0,
    "sambamba": 0.25,
    "manta": 8,
    "breakseq2": 0,
    "delly": 1,
    "lumpy": 1,
}

TIMEOUTS = {
    "breakseq2": "6h",
    "manta": "6h",
    "breakdancer_config": "2h",
    "breakdancer": "4h",
}
LUMPY_EXCLUDE_BED_FILENAME = "/resources/{genome}.bed"
CONTIGS_FILTER = "^hs37d5|alt|_random|_decoy"
MIN_CONTIG_LENGTH = 1000000
LOG_FILE_DIR = "./log/{tool}_log/"
BWA_PATH = "/usr/local/bin/bwa"
SAMTOOLS_PATH = "/usr/local/bin/samtools"
KNOWN_BREAKPOINT_FILENAME = {
    "hg19": None,
    "b37": "/breakseq2_bplib_20150129/breakseq2_bplib_20150129.gff",
    "hg38": None,
}
BREAKSEQ_WORK_DIR = "./breakseq2"
MANTA_WORK_DIR = "./manta"
BREAKDANCER_CFG_FILENAME = "breakdancer.cfg"
MIN_READS_FOR_CONTIG_ANALYSIS = 10


class WeightedPool:
    """This class provides a multiprocessing.Pool like interface with the
    added ability to specify the weight of a process to execute. The class
    is not thread-safe, so users should only use a given WeightedPool object
    in a single thread.
    """

    def __init__(self, max_weight=multiprocessing.cpu_count(), poll_interval=2):
        """Create a new WeightedPool object.

       Parameters
       ----------
       max_weight : float/int, optional
           The maximum weight for this given pool, by default the number of available
           cores.
       poll_interval : int, optional
           The number of seconds to wait when checking for completed processes in our
           process pool, by default 2
       """
        self._max_weight = max_weight
        self._poll_interval = poll_interval
        self._current_weight = 0
        self._current_pool = set()

    def _update_pool(self):
        """This method will look at all process currently in progress and check
        to see if any have completed. It will remove any completed process from the
        pool and will subtract their weights from the total weight of the pool.
        """
        # We'll keep track of which processes have completed and their weights.
        to_delete = set()
        completed_weight = 0

        for (process, weight, cmd, stdout, stderr) in self._current_pool:
            status = process.poll()
            # A status of None indicates the process is still running.
            if status is not None:
                to_delete.add((process, weight, cmd))
                completed_weight += weight
                if stdout:
                    stdout.close()
                if stderr:
                    stderr.close()
            # We'll warn if we received a non-zero exit code. Perhaps add a feature
            # later that would permit the user to choose to error instead?
            if status is not None and status != 0:
                logging.warning("Command {} exited with status {}.".format(cmd, status))
        self._current_pool -= to_delete
        self._current_weight -= completed_weight

    def _wait_for_weight(self, weight_target):
        """This method will block until the current pool has at most the stated
        weight target.

        Parameters
        ----------
        weight_target : float/int
            Block until our pool has at most this weight.
        """
        self._update_pool()
        while self._current_weight > weight_target:
            time.sleep(self._poll_interval)
            self._update_pool()

    def apply_async(self, cmd, weight=1, stdout=None, stderr=None):
        """This method will submit a process to our weighted pool. If there is not
        room in our pool, it will block until it can submit the process. This means
        there could be a deadlock condition if no jobs in the pool complete.

        Parameters
        ----------
        cmd : list[str]
            The command to be executed (will be passed to subprocess.Popen())
        weight : float/int, optional
            The weight for the given process. This is the size that will be taken up in
            our pool, by default 1.
        stdout : str, optional
            A filename to store stdout.
        stderr : str, optional
            A filename to store stderr.
        """
        # First, wait until we have room in our
        self._wait_for_weight(self._max_weight - weight)
        stdout = open(stdout, "w") if stdout else None
        stderr = open(stderr, "w") if stderr else None
        process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        self._current_weight += weight
        self._current_pool.add(
            (process, weight, subprocess.list2cmdline(cmd), stdout, stderr)
        )

    def join(self):
        """This method will wait until all processes in our pool are complete.
        """
        self._wait_for_weight(0)


def parse_arguments():
    """This function will parse the arguments to our program.

    Returns
    -------
    argparse.Namespace
        A structure with the parsed arguments.
    """
    args = argparse.ArgumentParser(description="Parliament2")
    args.add_argument(
        "--bam",
        required=True,
        help=(
            "The name of the Illumina BAM file for which to call structural "
            "variants containing mapped reads."
        ),
    )
    args.add_argument(
        "-r",
        "--ref_genome",
        required=True,
        help=(
            "The name of the reference file that matches the reference used "
            "to map the Illumina inputs."
        ),
    )
    args.add_argument(
        "--prefix",
        required=False,
        help=(
            "(Optional) If provided, all output files will start with this. If "
            "absent, the base of the BAM file name will be used."
        ),
    )
    args.add_argument(
        "--filter_short_contigs",
        action="store_true",
        help=(
            "If selected, SV calls will not be generated on contigs shorter than 1 MB."
        ),
    )
    args.add_argument(
        "--breakdancer",
        action="store_true",
        help="If selected, the program Breakdancer will be one of the SV callers run.",
    )
    args.add_argument(
        "--breakseq",
        action="store_true",
        help="If selected, the program BreakSeq2 will be one of the SV callers run.",
    )
    args.add_argument(
        "--manta",
        action="store_true",
        help="If selected, the program Manta will be one of the SV callers run.",
    )
    args.add_argument(
        "--cnvnator",
        action="store_true",
        help="If selected, the program CNVnator will be one of the SV callers run.",
    )
    args.add_argument(
        "--lumpy",
        action="store_true",
        help="If selected, the program Lumpy will be one of the SV callers run.",
    )
    args.add_argument(
        "--delly_deletion",
        action="store_true",
        help=(
            "If selected, the deletion module of the program Delly2 will be one of "
            "the SV callers run."
        ),
    )
    args.add_argument(
        "--delly_insertion",
        action="store_true",
        help=(
            "If selected, the insertion module of the program Delly2 will be one "
            "of the SV callers run."
        ),
    )
    args.add_argument(
        "--delly_inversion",
        action="store_true",
        help=(
            "If selected, the inversion module of the program Delly2 will be one "
            "of the SV callers run."
        ),
    )
    args.add_argument(
        "--delly_duplication",
        action="store_true",
        help=(
            "If selected, the duplication module of the program Delly2 will be one "
            "of the SV callers run."
        ),
    )
    args.add_argument(
        "--genotype",
        action="store_true",
        help=(
            "If selected, candidate events determined from the individual callers "
            "will be genotyped and merged to create a consensus output."
        ),
    )
    args.add_argument(
        "--svviz",
        action="store_true",
        help=(
            "If selected, visualizations of genotyped SV events will be produced "
            "with SVVIZ, one screenshot of support per event. For this option to "
            "take effect, Genotype must be selected."
        ),
    )
    args.add_argument(
        "--svviz_only_validated_candidates",
        action="store_true",
        help=(
            "Run SVVIZ only on validated candidates? For this option to be "
            "relevant, SVVIZ must be selected. NOT selecting this will make "
            "the SVVIZ component run longer."
        ),
    )
    args.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose logging information.",
    )

    parsed_args = args.parse_args()
    if parsed_args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    return parsed_args


def get_reference_genome_name(ref_genome_filename):
    """Sniff the given reference genome and make a decision about what
    human reference version it represents.

    Parameters
    ----------
    ref_genome_filename : str
        The filename for the reference genome FASTA

    Returns
    -------
    str
        A string stating whether the reference appears to be hg19, b37, or hg38.
    """

    # TODO The logic here looks very shaky. We may want to consider switching to
    # a more robust method.
    # The logic:
    #        1. If the contig names do not start with chr, then it is b37
    #        2. If we see a contig line for chr1 and LN:248956422 is in that line, then
    #           it is hg38
    #        3. Otherwise, if the contig names begin with chr and there are 195 contigs
    #           it is hg38, otherwise hg19

    number_of_contigs = 0
    with open(ref_genome_filename) as fh:
        for line in fh:
            if line.startswith(">"):
                if not line.startswith(">chr"):
                    return "b37"
                elif line.startswith(">chr1") and "LN:248956422" in line:
                    return "hg38"
                else:
                    number_of_contigs += 1

    if number_of_contigs == 195:
        return "hg38"

    return "hg19"


def get_contigs(bam_filename, filter_short_contigs):
    """Get a list of contigs that were used during mapping.

    Parameters
    ----------
    bam_filename : str
        The filename for mapped reads in BAM format.
    filter_short_contigs : bool
        Should we filter out shorter contigs?

    Returns
    -------
    list[str]
        A list of contigs that met any filtering criteria.
    """
    contigs = []

    samtools_proc = subprocess.Popen(
        ["samtools", "view", "-H", bam_filename], stdout=subprocess.PIPE
    )
    for line in samtools_proc.stdout:
        # Contig lines begin with @SQ
        if line[:3] != "@SQ":
            continue
        sequence_name = line.strip().split("SN:")[-1].split("\t")[0]
        length = int(line.strip().split("LN:")[-1].split("\t")[0])

        if filter_short_contigs and (
            re.findall(CONTIGS_FILTER, sequence_name) != []
            or length < MIN_CONTIG_LENGTH
        ):
            continue
        contigs.append(sequence_name)

    return contigs


def convert_cram_to_bam(cram_filename, ref_genome_filename):
    """Convert the given CRAM file to a BAM file along with a BAM index.

    Parameters
    ----------
    cram_filename : str
        The CRAM file
    ref_genome_filename : str
        The filename for the reference genome FASTA

    Returns
    -------
    str
        The path of the new BAM file.
    """
    # Save a core for indexing the BAM file.
    num_cores = max(1, multiprocessing.cpu_count() - 1)

    # Create a fifo to stream to so that we can create the index file while
    # we are performing the conversion. The fifo will have our final BAM
    # filename, and we'll use a temp file to store the converted BAM.
    _, temp_bam_filename = tempfile.mkstemp(suffix=".bam")
    bam_filename = os.path.splitext(cram_filename)[0] + ".bam"
    os.mkfifo(bam_filename)

    with open(temp_bam_filename, "w") as bam_fh:
        cmd = [
            "samtools",
            "view",
            cram_filename,
            "-bh",
            "-@",
            str(num_cores),
            "-T",
            ref_genome_filename,
            "-o",
            "-",
        ]
        convert_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        cmd = ["tee", bam_filename]
        tee_proc = subprocess.Popen(cmd, stdin=convert_proc.stdout, stdout=bam_fh)

        cmd = ["samtools", "index", bam_filename]
        subprocess.check_call(cmd)
        tee_proc.communicate()

    shutil.move(temp_bam_filename, bam_filename)

    return bam_filename


def gunzip_input(filename, overwrite=False):
    """Gunzip the given file.

    Parameters
    ----------
    filename : str
        The name of the gzipped file
    overwrite : bool, optional
        Should we overwrite the file if an uncompressed version exists?
        By default False.

    Returns
    -------
    str
        The uncompressed filename
    """
    prefix, suffix = os.path.splitext(filename)
    if suffix != ".gz":
        logging.warning(
            "The file {} lacks a .gz file extension. Will skip gunzip.".format(filename)
        )
        return filename

    if not os.path.isfile(prefix) or overwrite:
        subprocess.check_call(["gunzip", "-f", filename])

    return prefix


def skip_contig_analysis(contig, bam_filename):
    """Checks whether the given contig should be skipped in our SV analysis.
    Currently this check simply looks to see if there are some minimum number
    of reads that mapped to the given contig.

    Parameters
    ----------
    contig : str
        The contig name that we may want to analyze.
    bam_filename : str
        The filename for mapped reads in BAM format.

    Returns
    -------
    bool
        A boolean indicating whether the contig should be skipped because it
        failed to meet our criteria for analysis.
    """
    cmd = ["samtools", "view", contig]
    samtools_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for i, line in enumerate(samtools_proc.stdout):
        if i >= MIN_READS_FOR_CONTIG_ANALYSIS - 1:
            samtools_proc.kill()
            return False

    return True


def setup_logging_env(prefix, tool, contig=None):
    """Create the directory to hold output logs for this tool and 
    construct possible names for the stdout and stderr logs.

    Parameters
    ----------
    prefix : str
            The prefix to use for outputs.
    tool : str
        The name of the tool that will generate logs
    contig : str, optional
        An optional string indicating a particular contig that
        is being analyzed.

    Returns
    -------
    (str, str)
        The proposed filenames to hold stdout and sterr logs.
    """
    if contig:
        logging.info("Running {} for contig {}".format(tool, contig))
    else:
        logging.info("Running {}".format(tool))

    # Setup log directory
    tool = tool.lower()
    log_dirs = LOG_FILE_DIR.format(tool=tool)
    os.makedirs(log_dirs)

    # Create potential log filenames
    if contig:
        stdout_filename = os.path.join(
            log_dirs, "{}.{}.{}.stdout".format(prefix, tool, contig)
        )
        stderr_filename = os.path.join(
            log_dirs, "{}.{}.{}.stderr".format(prefix, tool, contig)
        )
    else:
        stdout_filename = os.path.join(log_dirs, "{}.{}.stdout".format(prefix, tool))
        stderr_filename = os.path.join(log_dirs, "{}.{}.stderr".format(prefix, tool))

    return (stdout_filename, stderr_filename)


def run_parliament(
    bam_filename,
    ref_genome_filename,
    prefix,
    filter_short_contigs,
    breakdancer,
    breakseq,
    manta,
    cnvnator,
    lumpy,
    delly_deletion,
    delly_insertion,
    delly_inversion,
    delly_duplication,
    genotype,
    svviz,
    svviz_only_validated_candidates,
):
    """Run parliament2.

        Parameters
        ----------
        bam_filename : str
            The filename of the bam/cram file to analyze.
        ref_genome_filename : str
            The filename of the reference genome.
        prefix : str
            The prefix to use for outputs.
        filter_short_contigs : bool
            Should we filter shorter contigs in our reference?
        breakdancer : bool
            Should we run breakdancer?
        breakseq : bool
            Should we run breakseq?
        manta : bool
            Should we run manta?
        cnvnator : bool
            Should we run cnvnator?
        lumpy : bool
            Should we run lumpy?
        delly_deletion : bool
            Should we run Delly in deletion mode?
        delly_insertion : bool
            Should we run Delly in insertion mode?
        delly_inversion : bool
            Should we run Delly in inversion mode?
        delly_duplication : bool
            Should we run Delly in duplication mode?
        genotype : bool
            Should we genotype samples and create a consensus VCF?
        svviz : bool
            Should we create visualization outputs?
        svviz_only_validated_candidates : bool
            Should we run only visualize validated candidates?
        """
    genome_name = get_reference_genome_name(ref_genome_filename)
    contigs = get_contigs(bam_filename, filter_short_contigs)
    pool = WeightedPool()

    if breakseq or manta:
        logging.info("Launching jobs that cannot be parallelized by contig")

    # BreakSeq
    if breakseq:
        stdout_filename, stderr_filename = setup_logging_env(prefix, "BreakSeq2")
        cmd = [
            "timeout",
            TIMEOUTS["breakseq2"],
            "/home/dnanexus/breakseq2-2.2/scripts/run_breakseq2.py",
            "--reference",
            ref_genome_filename,
            "--bams",
            bam_filename,
            "--work",
            BREAKSEQ_WORK_DIR,
            "--bwa",
            BWA_PATH,
            "--samtools",
            SAMTOOLS_PATH,
            "--nthreads",
            str(multiprocessing.cpu_count()),
            "--sample",
            prefix,
        ]
        # Note: in original Parliament2, the code always used the file
        # breakseq2_bplib_20150129.gff which is uses b37 coordinates. Currently,
        # for hg19 and hg38 the code will not add a gff file, but we may want
        # to find comparable gff files for those references or do a liftover.
        bplib_gff = KNOWN_BREAKPOINT_FILENAME[genome_name]
        if bplib_gff is not None:
            cmd.extend(["--bplib_gff", bplib_gff])
        else:
            logging.warning(
                "No known breakpoint file for {} genome.".format(genome_name)
            )

        pool.apply_async(
            cmd,
            SV_CALLER_WEIGHTS["breakseq2"],
            stdout=stdout_filename,
            stderr=stderr_filename,
        )

    if manta:
        stdout_filename, stderr_filename = setup_logging_env(prefix, "Manta")
        cmd = [
            "timeout",
            TIMEOUTS["manta"],
            "python",
            "/miniconda/bin/configManta.py",
            "--referenceFasta",
            ref_genome_filename,
            "--normalBam",
            bam_filename,
            "--runDir",
            MANTA_WORK_DIR,
        ]
        cmd += ["--region={}".format(contig) for contig in contigs]
        subprocess.check_call(cmd)

        # The original script had the number of cores hard-coded to 16. I'm assuming
        # that was under the assumption that it was being run on a 32 core machine?
        # I'm changing this logic so that it will use half of the total cores
        # available, but max out at 16. This can be changed if there are better and
        # more informed opinions.
        num_cores = min(multiprocessing.cpu_count() / 2, 16)
        cmd = [
            "timeout",
            TIMEOUTS["manta"],
            "python",
            os.path.join(MANTA_WORK_DIR, "runWorkflow.py"),
            "-m",
            "local",
            "-j",
            str(num_cores),
        ]
        pool.apply_async(
            cmd, weight=num_cores, stdout=stdout_filename, stderr=stderr_filename
        )

    # While Breakdancer will be run in multiple threads for each contig, we need to
    # create a config file first that will be used by each thread.
    if breakdancer:
        cmd = [
            "timeout",
            TIMEOUTS["breakdancer_config"],
            "/breakdancer/cpp/bam2cfg",
            "-o",
            BREAKDANCER_CFG_FILENAME,
            bam_filename,
        ]
        subprocess.check_call(cmd)

    #    lumpy_bed = LUMPY_EXCLUDE_BED_FILENAME.format(genome=get_reference_genome_name(ref_genome_filename))

    processed_contigs = []
    run_delly = any(
        [delly_deletion, delly_duplication, delly_insertion, delly_inversion]
    )
    if cnvnator or run_delly or breakdancer or lumpy:
        count = 0
        for contig in contigs:
            # Skip this contig if we determine it shouldn't be analyzed.
            if skip_contig_analysis(contig, bam_filename):
                continue
            processed_contigs.append(contig)
            if breakdancer:
                stdout_filename, stderr_filename = setup_logging_env(
                    prefix, "breakdancer", contig
                )
                cmd = [
                    "timeout",
                    TIMEOUTS["breakdancer"],
                    "/breakdancer/cpp/breakdancer-max",
                    BREAKDANCER_CFG_FILENAME,
                    bam_filename,
                    "-o",
                    contig,
                ]
                output_filename = "breakdancer-{}.ctx".format(contig)
                pool.apply_async(
                    cmd,
                    stdout=output_filename,
                    stderr=stderr_filename,
                    weight=SV_CALLER_WEIGHTS["breakdancer"],
                )

            if cnvnator:
                stdout_filename, stderr_filename = setup_logging_env(
                    prefix, "CNVnator", contig
                )
                cmd = ["runCNVnator", contig, str(count)]
                pool.apply_async(
                    cmd,
                    stdout=stdout_filename,
                    stderr=stderr_filename,
                    weight=SV_CALLER_WEIGHTS["cnvnator"],
                )

            count += 1


def main():
    args = parse_arguments()

    if args.prefix is None:
        args.prefix = os.path.splitext(args.bam)[0]
    args.run_delly = any(
        [
            args.delly_deletion,
            args.delly_insertion,
            args.delly_inversion,
            args.delly_duplication,
        ]
    )
    args.ref_genome = gunzip_input(args.ref_genome)

    # Check that at least one of the SV callers was set to run.
    if not any(
        [
            args.breakdancer,
            args.breakseq,
            args.manta,
            args.cnvnator,
            args.lumpy,
            args.run_delly,
        ]
    ):
        logging.warning(
            (
                "Did not detect any SV modules requested by the user through "
                "command-line flags."
            )
        )
        logging.warning(
            (
                "Running with default SV modules: Breakdancer, Breakseq, Manta, "
                "CNVnator, Lumpy, and Delly Deletion"
            )
        )
        args.breakdancer = True
        args.breakseq = True
        args.manta = True
        args.cnvnator = True
        args.lumpy = True
        args.delly_deletion = True
        args.run_delly = True

    # Check if the input bam is actually a cram file. If it is, we'll create
    #  a bam version as well.
    if os.path.splitext(args.bam)[-1].lower() == ".cram":
        args.bam = convert_cram_to_bam(args.bam)

    # Check that we have the bam and fasta index files.
    if not os.path.isfile(args.bam + ".bai"):
        subprocess.check_call(["samtools", "index", args.bam])
    if not os.path.isfile(args.ref_genome_name + ".fai"):
        subprocess.check_call(["samtools", "faidx", args.ref_genome_name])

    run_parliament(
        args.bam,
        args.ref_genome,
        args.prefix,
        args.filter_short_contigs,
        args.breakdancer,
        args.breakseq,
        args.manta,
        args.cnvnator,
        args.lumpy,
        args.delly_deletion,
        args.delly_insertion,
        args.delly_inversion,
        args.delly_duplication,
        args.genotype,
        args.svviz,
        args.svviz_only_validated_candidates,
    )


if __name__ == "__main__":
    main()
