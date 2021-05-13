import sys 
def parse_line(line):
    return int(line.split()[8])

print(sum([parse_line(line) for line in open(sys.argv[1]).readlines()[2:]])/10**9)