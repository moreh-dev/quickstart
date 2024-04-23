# Usage
## 0. Setup
현재는 파일간의 경로를 간이로 설정해둬서 해당 폴더 내의 파일을 모두 ~/quickstart 로 이동시켜주세요.

## 1. Scalability test
Tmux를 열어 `test_scalibality.sh` 를 실행해주세요.
스크립트에 필요한 인자는 두가지 입니다.
1. model_name : train_${model}.py 에 해당하는 model 이름을 사용하세요. (현재는 mistral, gpt, baichuan만 작성해두었습니다.)
2. max_batch_size : xlarge에서 가장 gpu memory 를 많이 사용하는 batchsize를 입력하세요.(만약 xlarge에서 OOM이고 2xlarge 기준이면, 2로 나누어서 입력해주세요)

### 주의사항
train_model.py 내부에서 parser에 아래와 같은 argument를 명시해주시고, 코드내에서 입력이 될 수 있게 작성해주세요
batch size : `--batch-size`
block size(seq_len) : `--block-size`
epoch : `--epochs`

## 2. Moreh-smi logging
mem usage를 파악하기 위함입니다.
Tmux의 다른 Window 혹은 다른 Pane에서 `get_moreh_log.sh`를 실행하세요

### 주의사항
해당 스크립트는 중단되지 않음으로 사용 종료 후 직접 종료해주세요. (tmux를 종료할 경우 알아서 종료될 것이니 신경안쓰셔도 됩니다)

## 3. Parsing
위에서 실행한 스크립트의 결과는 모두 logs/ 에 모이게 됩니다.
`parse.sh` 를 실행하면 SDA와 batch size에 따른 max throughput, max mem usage가 출력됩니다. 

