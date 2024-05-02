HOME=/home/ubuntu/
pip install tokenizers==0.11.6 
pip install transformers==4.17.0
pip install rich[jupyter]
cd $HOME; curl -L https://www.dropbox.com/sh/pfg8j6yfpjltwdx/AAC8Oky0w8ZS-S3S5zSSAuQma?dl=1 > mrqa-few-shot.zip; unzip mrqa-few-shot.zip -d mrqa-few-shot
