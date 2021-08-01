FROM stablebaselines/rl-baselines3-zoo

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN mkdir log
RUN mkdir params

CMD ["tensorboard" , "--logdir=log" , "--bind_all"]