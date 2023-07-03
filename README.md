# pre_XBE
Codes for pre-training KG encoder for XBE model.

# Pre-training Steps
1. Prepare Knowledge Grpah(KG) triplets (e.g., `kg.txt`) and a dictionary for indexing KG elements (i.e., entities and relations) (e.g., `sym2id.json`).
 - `./data/kg.txt`
   ~~~

   ~~~
- `./data/sym2id.json`
  ~~~

  ~~~
2. Pre-training the KG encoder via following command:
   ~~~
   python main_kg.py \
       --gpu 1 \
       --batch_size 50 \
       --epoch 10 \
       --data_addr data/kg.txt \
       --sym2id_addr data/sym2id.json
   ~~~
3. Put the pre-trained checkpoint (e.g., *.ckpt) under `code_xbe/xbe/ckpt_kg`.
