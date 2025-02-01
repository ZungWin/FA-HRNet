#!/bin/bash

LOG_FILE="test_coco.log"

> $LOG_FILE

# loop through
for epoch in {209..1}
do
  echo "Running test for epoch: $epoch"

  # execute the Python script, temporarily store the output in temp_output.log
  python test.py --gpu {gpu_id} --cfg {your_coco_yml_path} --exp_dir ../output/exp_{month}-{date}_{hour}:{minute} --test_epoch $epoch > temp_output.log 2>&1

  # extract the COCO validation results, save them to a log file
  echo "Epoch $epoch" >> $LOG_FILE
  grep "Average Precision\|Average Recall" temp_output.log >> $LOG_FILE
  echo "" >> $LOG_FILE

done

# clean up temporary files
rm temp_output.log

echo "All tests completed. Results saved to $LOG_FILE"
