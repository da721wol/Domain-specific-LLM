import argparse
from transformers import FlaxT5ForConditionalGeneration


tf_model = FlaxT5ForConditionalGeneration.from_pretrained("t5-efficient-gc4-german-base-nl36", from_pt=True)
tf_model.save_pretrained("data/t5-german")