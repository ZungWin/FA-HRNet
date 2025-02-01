#!/bin/bash

LOG_FILE="test_mpii.log"

> $LOG_FILE

# loop through
for epoch in {209..1}
do
    # execute the Python script, temporarily store the output in temp_output.log
    python test.py --gpu {gpu_id} --cfg {your_coco_yml_path} --exp_dir ../output/exp_{month}-{date}_{hour}:{minute} --test_epoch $epoch 2>/dev/null | grep "OrderedDict")

    # extract the MPII test results
    mean=$(echo $result | grep -oP "(?<=('Mean', )).+?(?=, \('Mean@0.1')")

    # save them to a log file
    echo "Epoch $epoch: $result" >> test.log

done

echo "All tests completed. Results saved to $LOG_FILE"
