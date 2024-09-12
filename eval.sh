#!/bin/bash

for DATASET in imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101
do
# python parse_test_res.py   loss_4_0/base2new/train_base/${DATASET}/shots_16_4.0/TCP/vit_b32_ep50_ctxv1/ --test-log
# python parse_test_res.py   loss_4_0/base2new/test_new/${DATASET}/shots_16_4.0/TCP/vit_b32_ep50_ctxv1/ --test-log
# python parse_test_res.py   loss_4_0/base2new/train_base/${DATASET}/shots_16_4.0/TCP/vit_b128_ep10_ctxv1/ --test-log
# python parse_test_res.py   loss_4_0/base2new/test_new/${DATASET}/shots_16_4.0/TCP/vit_b128_ep10_ctxv1/ --test-log

# python parse_test_res.py   modal_4_0/base2new/train_base/${DATASET}/0shots_16_4.0/TCP/vit_b128_ep10_ctxv1 --test-log
# python parse_test_res.py   modal_4_0/base2new/test_new/${DATASET}/0shots_16_4.0/TCP/vit_b128_ep10_ctxv1 --test-log
# python parse_test_res.py output_0617_cd/evaluation/TCP/vit_b128_ep5_ctxv1_cross_dataset_16shots/${DATASET} --test-log
# python parse_test_res.py   output_0616_fsl_4_0/base2new/train_base/${DATASET}/shots_4_4.0/TCP/vit_b128_ep10_ctxv1/ --test-log
python parse_test_res.py   weight_4_0/base2new/train_base/${DATASET}/116_4.0/TCP/vit_b32_ep50_ctxv1/ --test-log
python parse_test_res.py   weight_4_0/base2new/test_new/${DATASET}/116_4.0/TCP/vit_b32_ep50_ctxv1/ --test-log
# python parse_test_res.py   weight_4_0/base2new/train_base/${DATASET}/1_shots_16_4.0/TCP/vit_b128_ep10_ctxv1/ --test-log
# python parse_test_res.py   weight_4_0/base2new/test_new/${DATASET}/1_shots_16_4.0/TCP/vit_b128_ep10_ctxv1/ --test-log
done
