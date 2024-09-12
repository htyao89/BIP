#!/bin/bash

cd ..

# custom config
DATA=/hy-tmp/data
TRAINER=TCP
WEIGHT=4.0

CFG=vit_b32_ep50_ctxv1
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
FOLDER=output_0627
#cfg.TRAINER.COOP.VFW = 0.75
#cfg.TRAINER.COOP.TFW = 1.0
#cfg.TRAINER.COOP.TL = 1
#cfg.TRAINER.COOP.VL = 1

VFW=0.75
TFW=1.0
L=0

#for L in 0 1 2 3 4 5 6 7 8 9 10
#do
#L=0
#for WEIGHT in 4.0 2.0 6.0 8.0
#do
for TFW in 1.0
do
for VFW in 0.5 0.6 0.7 0.8 0.9 1.0
do

CFG=vit_b32_ep50_ctxv1

for DATASET in eurosat dtd fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101 caltech101
do
for SEED in 1 2 3
do
    DIR=${FOLDER}_${NCTX}_${L}_${VFW}_${TFW}/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.TL ${L} \
        TRAINER.COOP.VL ${L} \
        TRAINER.COOP.VFW ${VFW}\
        TRAINER.COOP.TFW ${TFW}\
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done


LOADEP=50
SUB=new
for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/seed${SEED}
    MODEL_DIR=${FOLDER}_${NCTX}_${L}_${VFW}_${TFW}//base2new/train_base/${COMMON_DIR}
    DIR=${FOLDER}_${NCTX}_${L}_${VFW}_${TFW}//base2new/test_${SUB}/${COMMON_DIR}

    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.TL ${L} \
        TRAINER.COOP.VL ${L} \
        TRAINER.COOP.VFW ${VFW}\
        TRAINER.COOP.TFW ${TFW}\
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}\
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
done


CFG=vit_b128_ep10_ctxv1

for DATASET in sun397
do
for SEED in 1 2 3
do
    DIR=${FOLDER}_${NCTX}_${L}_${VFW}_${TFW}/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.L ${L} \
        TRAINER.COOP.TL ${L} \
        TRAINER.COOP.VL ${L} \
        TRAINER.COOP.VFW ${VFW}\
        TRAINER.COOP.TFW ${TFW}\
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done


LOADEP=25
SUB=new
for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/seed${SEED}
    MODEL_DIR=${FOLDER}_${NCTX}_${L}_${VFW}_${TFW}/base2new/train_base/${COMMON_DIR}
    DIR=${FOLDER}_${NCTX}_${L}_${VFW}_${TFW}//base2new/test_${SUB}/${COMMON_DIR}

    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.TL ${L} \
        TRAINER.COOP.VL ${L} \
        TRAINER.COOP.VFW ${VFW}\
        TRAINER.COOP.TFW ${TFW}\
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
done

done
done

