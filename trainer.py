print('[+] Preparing dataset..')
from datasets import load_dataset, concatenate_datasets
dataset = load_dataset('0xMaka/trading-candles-subset-sc-format')#, split='train[:50%]').train_test_split()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True), batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -format columns
tokenized_dataset = tokenized_dataset.remove_columns('text')
tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
tokenized_dataset.set_format('torch')
print(f'[+] Column names: {tokenized_dataset["train"].column_names}')

# - define dataloaders
BATCH_SIZE = 256
WORKERS = 20
print(f'[+] Setting up dataloader..\n[-] Batch size: {BATCH_SIZE}\n[-] Number of workers: {WORKERS}')
from torch.utils.data import DataLoader
train_dataloader = DataLoader(
  tokenized_dataset['train'], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator, pin_memory=True, num_workers=WORKERS
)

from torch.utils.data import DataLoader
eval_dataloader = DataLoader(
  tokenized_dataset['test'], batch_size=BATCH_SIZE, collate_fn=data_collator, pin_memory=True, num_workers=WORKERS
)
for batch in train_dataloader:
  break
print('[-] Batch: ')
print({k: v.shape for k, v in batch.items()})
id2label =  { 0: 'Bearish', 1: 'Bullish' }
label2id =  { 'bearish': 0, 'Bullish': 1 }

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2, id2label=id2label, label2id=label2id)
outputs = model(** batch)
print(f'[-] Output Loss and and Logit Shape: {outputs.loss} | {outputs.logits.shape}')

# - pre train
print('[+] Setting optimizer..')
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

print('[+] Setting up scheduler..')
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
  'linear',
  optimizer=optimizer,
  num_warmup_steps=0,
  num_training_steps=num_training_steps,
)
print (f'[-] Number of training steps: {num_training_steps}')

print('[+] Moving model to device..')
from torch import device, backends
backends.cuda.matmul.allow_tf32 = True
model.to(device('cuda'))

#- train
import evaluate
accuracy = evaluate.load('accuracy')
print('[+] Training...')
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
  for batch in train_dataloader:
    batch = {k: v.to(device('cuda')) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)

print('[+] Evaluating model..')
from torch import argmax, no_grad
model.eval()
for batch in eval_dataloader:
  batch = {k: v.to(device('cuda')) for k, v in batch.items()}
  with no_grad():
    outputs = model(** batch)

  logits = outputs.logits
  predictions = argmax(logits, dim=1)
  accuracy.add_batch(predictions=predictions, references=batch['labels'])

print(accuracy.compute())

OUT='based-bert-sc'
tokenizer.save_pretrained(OUT)
model.save_pretrained(OUT)
