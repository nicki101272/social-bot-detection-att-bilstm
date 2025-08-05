# Social Bot Detection using Att-BiLSTM and GAN

This project is part of my master's thesis research and explores contextual social bot detection using deep learning methods.

## 🔍 Summary

- Uses an attention-based BiLSTM model to classify social bots and humans based on tweet content.
- Compares performance with BERT-based models.
- Applies Seq-GAN for data augmentation to address class imbalance.
- Evaluated on the Cresci dataset (2017).

## 📁 Contents

- `main_model.py`: Implementation of Att-BiLSTM
- `seq_gan.py`: GAN-based data augmentation module
- `preprocessing/`: Scripts for text preprocessing
- `results/`: Output metrics and performance plots
- `report.pdf`: Final manuscript

## 📊 Dataset

We used the publicly available [Cresci Bot Dataset (2017)](https://github.com/dfreelon/cresci-2017).

## 🧠 Model Highlights

- Attention layer improves interpretability
- Seq-GAN significantly enhances classifier performance
- Outperforms classical ML baselines and matches BERT in certain settings

## 📄 Paper

You can read the full paper [here](./report.pdf)

## 📫 Contact

**Nicki Sadeghi**  
nicki.sadeqi@gmail.com
