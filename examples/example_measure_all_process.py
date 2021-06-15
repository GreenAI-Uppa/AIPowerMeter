import measure_utils 

outfile = 'result.json'
process_tree = measure_utils.get_pids()
p, q = measure_utils.measure_from_pid_list(list(process_tree.nodes), outfile=outfile, period=10)
print('measuring how you burn the planet')
while True:
    continue
