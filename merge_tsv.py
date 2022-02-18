import csv
import sys
csv.field_size_limit(sys.maxsize)
import argparse

if __name__ == '__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Merge tsv files (e.g., tmp0.csv, tmp1.csv, ...)')
    parser.add_argument('--num_gpus',  help='Total number of GPUs in the system',
                        default=4, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()


    test = ['tmp%d.csv'%(i) for i in range(args.num_gpus)]
    FIELDNAMES =  ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    outfile = 'merged.tsv'
    with open(outfile, 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)

        for infile in test:
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    try:
                      writer.writerow(item)
                    except Exception as e:
                      print(e)
