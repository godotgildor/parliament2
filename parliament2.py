#!/usr/bin/env python

import argparse
import subprocess
import logging
import multiprocessing
import os
import time


SV_CALLER_WEIGHTS = {
    'breakdancer': 1.0,
    'cnvnator': 1.0,
    'sambamba': 0.25,
    'manta': 8,
    'breakseq': 0,
    'delly': 1,
    'lumpy': 1
}

class WeightedPool():
    """This class provides a multiprocessing.Pool like interface with the
    added ability to specify the weight of a process to execute. The class
    is not thread-safe, so users should only use a given WeightedPool object
    in a single thread.
    """

    def __init__(self, max_weight=multiprocessing.cpu_count(), poll_interval=2):
        """Create a new WeightedPool object.

        Args:
            max_weight (float/int): The maximum weight for this given pool.
            poll_interval (int): The number of seconds to wait when checking
                for completed processes in our process pool.
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

        for (process, weight, cmd) in self._current_pool:
            status = process.poll()
            # A status of None indicates the process is still running.
            if status is not None:
                to_delete.add((process, weight, cmd))
                completed_weight += weight
            # We'll warn if we received a non-zero exit code. Perhaps add a feature
            # later that would permit the user to choose to error instead?
            if status is not None and status != 0:
                logging.warning('Command {} exited with status {}.'.format(cmd, status))
        self._current_pool -= to_delete
        self._current_weight -= completed_weight


    def _wait_for_weight(self, weight_target):
        """This method will block until the current pool has at most the stated
        weight target.

        Args:
            weight_target (float/int): Block until our pool has at most this weight.
        """
        self._update_pool()
        while self._current_weight > weight_target:
            time.sleep(self._poll_interval)
            self._update_pool()


    def apply_async(self, cmd, weight=1, stdout=None, stderr=None):
        """This method will submit a process to our weighted pool. If there is not
        room in our pool, it will block until it can submit the process. This means
        there could be a deadlock condition if no jobs in the pool complete. 

        Args:
            cmd (list of strings): The command to be executed (will be passed to 
                subprocess.Popen())
            weight (float/int): The weight for the given process. This is the size that
                will be taken up in our pool.
            stdout (File handler): A file handler to store the stdout.
            stderr (File handler): A file handler to store the stderr.
        """
        # First, wait until we have room in our 
        self._wait_for_weight(self._max_weight - weight)
        process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        self._current_weight += weight
        self._current_pool.add((process, weight, subprocess.list2cmdline(cmd)))


    def join(self):
        """This method will wait until all processes in our pool are complete.
        """
        self._wait_for_weight(0)
            

def parse_arguments():
    """This function will parse the arguments to our program.

    Returns:
        A structure with the parsed arguments.
    """
    args = argparse.ArgumentParser(description='Parliament2')
    args.add_argument('--bam', required=True, help="The name of the Illumina BAM file for which to call structural variants containing mapped reads.")
    args.add_argument('-r', '--ref_genome', required=True, help="The name of the reference file that matches the reference used to map the Illumina inputs.")
    args.add_argument('--prefix', required=False, help="(Optional) If provided, all output files will start with this. If absent, the base of the BAM file name will be used.")
    args.add_argument('--filter_short_contigs', action="store_true", help="If selected, SV calls will not be generated on contigs shorter than 1 MB.")
    args.add_argument('--breakdancer', action="store_true", help="If selected, the program Breakdancer will be one of the SV callers run.")
    args.add_argument('--breakseq', action="store_true", help="If selected, the program BreakSeq2 will be one of the SV callers run.")
    args.add_argument('--manta', action="store_true", help="If selected, the program Manta will be one of the SV callers run.")
    args.add_argument('--cnvnator', action="store_true", help="If selected, the program CNVnator will be one of the SV callers run.")
    args.add_argument('--lumpy', action="store_true", help="If selected, the program Lumpy will be one of the SV callers run.")
    args.add_argument('--delly_deletion', action="store_true", help="If selected, the deletion module of the program Delly2 will be one of the SV callers run.")
    args.add_argument('--delly_insertion', action="store_true", help="If selected, the insertion module of the program Delly2 will be one of the SV callers run.")
    args.add_argument('--delly_inversion', action="store_true", help="If selected, the inversion module of the program Delly2 will be one of the SV callers run.")
    args.add_argument('--delly_duplication', action="store_true", help="If selected, the duplication module of the program Delly2 will be one of the SV callers run.")
    args.add_argument('--genotype', action="store_true", help="If selected, candidate events determined from the individual callers will be genotyped and merged to create a consensus output.")
    args.add_argument('--svviz', action="store_true", help="If selected, visualizations of genotyped SV events will be produced with SVVIZ, one screenshot of support per event. For this option to take effect, Genotype must be selected.")
    args.add_argument('--svviz_only_validated_candidates', action="store_true", help="Run SVVIZ only on validated candidates? For this option to be relevant, SVVIZ must be selected. NOT selecting this will make the SVVIZ component run longer.")
    args.add_argument('--verbose', '-v', action="store_true", help="Print verbose logging information.")

    parsed_args = args.parse_args()
    if parsed_args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    return parsed_args


def gunzip_input(filename, overwrite=False):
    """Gunzip the given file.

    Args:
        filename (string): The name of the gzipped file
        overwrite (boolean): Should we overwrite the file if an uncompressed version exists?

    Returns:
        A string of giving the uncompressed filename
    """
    prefix, suffix = os.path.splitext(filename)
    if suffix != '.gz':
        logging.warning('The file {} lacks a .gz file extension. Will skip gunzip.'.format(filename))
        return filename
    
    if not os.path.isfile(prefix) or overwrite:
        subprocess.check_call(['gunzip', '-f', filename])
    
    return prefix


def run_parliament(bam_filename, 
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
                   svviz_only_validated_candidates):
    """Run parliament2.

    Args:
        bam_filename (string): The filename of the bam/cram file to analyze.
        ref_genome_filename (string): The filename of the reference genome.
        prefix (string): The prefix to use for outputs.
        filter_short_contigs (boolean): Should we filter shorter contigs in our reference?
        breakdancer (boolean): Should we run breakdancer?
        breakseq (boolean): Should we run breakseq?
        manta (boolean): Should we run manta?
        cnvnator (boolean): Should we run cnvnator?
        lumpy (boolean): Should we run lumpy?
        delly_deletion (boolean): Should we run Delly in deletion mode?
        delly_insertion (boolean): Should we run Delly in insertion mode?
        delly_inversion (boolean): Should we run Delly in inversion mode?
        delly_duplication (boolean): Should we run Delly in duplication mode?
        genotype (boolean): Should we genotype samples and create a consensus VCF?
        svviz (boolean): Should we create visualization outputs?
        svviz_only_validated_candidates (boolean): Should we run only visualize validated candidates?
    """
    foo = 1


def main():
    args = parse_arguments()
    
    if args.prefix is None:
        args.prefix = os.path.splitext(args.bam)[0]
    args.run_delly = any([args.delly_deletion, args, args.delly_insertion, args.delly_inversion, args.delly_duplication])
    args.ref_genome = gunzip_input(args.ref_genome)

    # Check that we have the bam and fasta index files.
    if not os.path.isfile(args.bam + '.bai'):
        subprocess.check_call(['samtools', 'index', args.bam])
    if not os.path.isfile(args.ref_genome_name + '.fai'):
        subprocess.check_call(['samtools', 'faidx', args.ref_genome_name])

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
        args.svviz_only_validated_candidates
    )


if __name__ == '__main__':
    main()
