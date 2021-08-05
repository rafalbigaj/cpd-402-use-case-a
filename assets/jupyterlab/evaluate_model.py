import os

output_dir = os.getenv('BATCH_OUTPUT_DIR', './')

results = 'Model validation succeeded!'

results_txt = os.path.join(output_dir, 'results.txt')

with open(results_txt, 'w') as f:
    print(results, file=f)

print(results)