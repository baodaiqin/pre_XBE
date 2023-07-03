# pre_XBE
Codes for pre-training KG encoder for XBE model.

# Pre-training Steps
1. Prepare Knowledge Grpah(KG) triplets (e.g., `kg.txt`) and a dictionary for indexing KG elements (i.e., entities and relations) (e.g., `sym2id.json`), which should be consistent with the one used in XBE model.
 - `./data/kg.txt`
   ~~~
   m.07ssc /common/phone_number/service_location   m.01fw9h
   m.0jm6n /sports/professional_sports_team/draft_picks    m.0cymln
   m.019rd /business/employment_tenure/person      m.01xcgf
   ~~~
- `./data/sym2id.json`
  ~~~
  {"m.01kyln": 0, "m.02pt7p4": 1, ... "/basketball/basketball_player/player_statistics": 21, ... "[MASK]": ...}
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
3. Put the pre-trained checkpoint (e.g., `kg_encoder.ckpt`) under `code_xbe/xbe/ckpt_kg`.
