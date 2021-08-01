FROM stablebaselines/rl-baselines3-zoo

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html



RUN mkdir log
RUN mkdir params

CMD ["tensorboard" , "--logdir=log" , "--bind_all"]