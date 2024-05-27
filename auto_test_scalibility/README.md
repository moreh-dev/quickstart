# Auto Test Script
파일 `auto_test.sh` 는 quickstart 레포의 코드들을 자동으로 테스트하기 위해서 만든 스크립트입니다.
이 스크립트를 통해 모델 및 하이퍼파라미터의 조합에 따른 throughput, vram 사용량을 확인할 수 있습니다.

# Usage
## 1. Write config file
`config.txt` 에 테스트하실 모델명과 argument 조합을 선언해주세요.
- `model_name` : 테스트할 모델명. quickstart repository에서 prepare_${model_name}_dataset.py, train_${model_name}.py, requirements_${model_name} 와 같이 이름의 규칙이 정해져있음을 가정하고 받는 Input입니다.
- `script_path` : 실행할 python script 파일 위치
- `model_path` : 모델의 체크포인트 위치 (테스트를 위함), default으로 줄 경우 코드 자체의 default path를 사용
- `log_path` : 실행결과에 대한 로그가 저장될 위치 디렉토리.
- `model_arguments` : Python script 실행 시 줄 argument들, 현재는 batch_size, block_size, sda를 넣을 수 있음.(quickstart repo에서 block_size는 조절 불가. 이는 다른 모델 테스트를 위해 추가한 기능)

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
}
]
```

}

## 2. Scalability test
Tmux를 열어 `auto_test.sh` 를 실행해주세요.
스크립트에 필요한 인자는 한가지 입니다.
1. FILE_NAME : config file에 해당하는 path를 넣어주세요

### 주의사항
train_${model}.py 내부에서 parser에 아래와 같은 argument를 명시해주시고, 코드내에서 입력이 될 수 있게 작성해주세요
batch size : `--batch-size`
block size(seq_len) : `--block-size`
epoch : `--epochs`

## 3. Parsing
위에서 실행한 스크립트의 결과는 모두 logs/ 에 모이게 됩니다.  
파일의 형식은 아래와 같습니다.  
  
`${log_path}/${model_name}_${sda}_batch${batch_size}_block${block_size}.log`  
`${log_path}/${model_name}_${sda}_batch${batch_size}_block${block_size}_moreh_smi.log`  
  
전자는 모델 학습의 로그, 후자는 학습 동안의 moreh-smi 로그입니다.  
`parse.sh` 를 실행하면 SDA와 batch size에 따른 max throughput, max mem usage가 출력됩니다. 

출력 예시)
<img width="784" alt="image" src="https://github.com/moreh-dev/quickstart/assets/138426917/3f13ae7d-6d1c-420f-8d4e-f1845dc86b2c">
