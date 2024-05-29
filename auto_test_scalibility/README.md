# Auto Test Script
파일 `auto_test.sh` 는 quickstart 레포의 코드들을 자동으로 테스트하기 위해서 만든 스크립트입니다.
이 스크립트를 통해 모델 및 하이퍼파라미터의 조합에 따른 throughput, vram 사용량을 확인할 수 있습니다.
측정은 테스트 케이스 당 30분을 기준으로 합니다.
지원 모델을 우선적으로 수정없이 llama3 70b 를 제외한 tutorial 모델이며, 타 모델 사용 시 아래의 설명대로 수정 후 사용 바랍니다.

## Tip : Different usage
만약 quickstart가 아닌 다른 곳에서 이 스크립트를 사용하고자 한다면 `auto_test.sh` 파일 내의 `morehdocs` 가 적힌 주석을 찾아 아래 두라인을 삭제하고 사용할 것.



# Usage

## 1. Write config file
`config.txt` 에 테스트하실 모델명과 argument 조합을 선언해주세요.

### Essential  
- `model_name` : (str) 테스트할 모델명. quickstart repository에서 prepare_${model_name}_dataset.py, train_${model_name}.py, requirements_${model_name} 와 같이 이름의 규칙이 정해져있음을 가정하고 받는 Input입니다.
- `script_path` : (str) 실행할 python script 파일 위치
- `model_path` : (str) 모델의 체크포인트 위치 (테스트를 위함), default으로 줄 경우 코드 자체의 default path를 사용
- `model_arguments` : (str) Python script 실행 시 줄 argument들, 현재는 batch_size, block_size, sda를 넣을 수 있음.(quickstart repo에서 block_size는 조절 불가. 이는 다른 모델 테스트를 위해 추가한 기능)
### Optional  
- `time_out`: (int or 'false') Python script를 종료할 시점. 기본값은 30 입니다.(분 단위). 'false'로 둘 경우 학습이 끝까지 진행됩니다.
- `log_path` : (str) 실행결과에 대한 로그가 저장될 위치 디렉토리.
### Example

```
Test_list = [
{
    model_name = "llama3",
    script_path = "tutorial/train_llama3.py",
    model_path = "/nas/team_cx/checkpoints/llama3-8b-base",
    log_path = "logs",
    model_arguments = [
        batch_size = default, block_size = default, sda=8,
        batch_size = 64, block_size = 1024, sda=6
    ]
},
{
    model_name = "llama2",
    script_path = "tutorial/train_llama2.py",
    model_path = "/nas/team_cx/checkpoints/llama2-13b-hf",
    log_path = "logs",
    model_arguments = [
        batch_size = default, block_size = default, sda=8
    ]
},
{
    time_out = false,
    model_name = "llama3",
    script_path = "tutorial/train_llama3.py",
    model_path = "/nas/team_cx/checkpoints/llama3-8b-base",
    log_path = "logs",
    model_arguments = [
        batch_size = default, block_size = default, sda=8,
        batch_size = 64, block_size = 1024, sda=6
    ]
},
]
```

### TIP
테스트시 qwen, mistral, baichuan 모델은 학습 데이터 크기가 작기 때문에 `time_out = 15` 로 두시면 학습 끝나고 모델이 저장되는 것을 방지할 수 있습니다.

## 2. Run Auto Test Script
`auto_test.sh` 를 ~/quickstart 위치로 복사 후 아래와 같이 실행시켜주세요.
이때 tmux를 사용하는 것을 추천드립니다.

```
~/quickstart$ bash auto_test.sh ${config_file_path}
```
스크립트에 필요한 인자는 한가지 입니다.
1. FILE_NAME : config file에 해당하는 path를 넣어주세요


### 주의사항
train_${model}.py 내부에서 parser에 아래와 같은 argument를 명시해주시고, 코드내에서 입력이 될 수 있게 작성해주세요
batch size : `--batch-size`
block size(seq_len) : `--block-size`
epoch : `--epochs`

## 3. Parsing
위에서 실행한 스크립트의 결과는 모두 default로 logs/ 에 모이게 됩니다.  
파일의 형식은 아래와 같습니다.  
  
`${log_path}/${model_name}_sda_${sda}_batch${batch_size}_block${block_size}.log`  
`${log_path}/${model_name}_sda_${sda}_batch${batch_size}_block${block_size}_moreh_smi.log`  
  
전자는 모델 학습의 로그, 후자는 학습 동안의 moreh-smi 로그입니다.  
`log_parser.py` 를 ~/quickstart 위치로 복사 후 실행하면 SDA와 batch size에 따른 max throughput, max mem usage가 출력됩니다. 
```
python log_parser.py
```

출력 예시)  
<img width="784" alt="image" src="https://github.com/moreh-dev/quickstart/assets/138426917/3f13ae7d-6d1c-420f-8d4e-f1845dc86b2c">

