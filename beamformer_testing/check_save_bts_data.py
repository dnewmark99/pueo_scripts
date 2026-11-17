import os

if __name__ == "__main__":
    run_ranges = [(1525, 1654)]
    chunks = [10]

    missing = []

    array = 0
    for r in range(len(run_ranges)):
        run_start = run_ranges[r][0]
        run_end = run_ranges[r][1]
        chunk_size = chunks[r]

        for run in range(run_start, run_end + 1):
            for chunk in range(chunk_size):

                saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/beamformerdata_run{run}_chunk{chunk}.json' 
                
                if not os.path.exists(saving_path):
                    missing.append((run, chunk, array))
                
                array += 1

    if missing:
        print('Missing JSON files:')
        for run, chunk, array in missing:
            print(f'  run={run}, chunk={chunk}, array={array}')
    else:
        print('Every run/chunk has a JSON file!')
    
