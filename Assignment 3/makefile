imagenet:
	test -f vgg_16.ckpt || \
	(wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz && \
	tar zxvf vgg_16_2016_08_28.tar.gz)
	python3 hw3_imagenet.py
uiuc:
	test -d pretrain || \
	(wget https://doc-04-0g-docs.googleusercontent.com/docs/securesc/limh8shadikorlbsnn6ggteesbl2ichj/vvnmove9a0rvnk80gne9h615tbrhknug/1527033600000/15756313507313528699/16786248718901200424/18BYtTJv0fwhJX9Y1vfxsrwPQh6nBSQGp?e=download&nonce=1mmsc5886gj5m&user=16786248718901200424&hash=uqo8brqo92kj6o0opsdcm70soo60c8cl && \
	unzip pretrain.zip)
	python3 hw3.py
