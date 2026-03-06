# QLASS
### 🛠️ Set up environment

Для воспроизведения экспериментов я настроил 2 окружения: одно использовал для инференса моделей, второе - для обучения. Все эксперименты проводились на бенчмарке Alfworld.  
Зависимости окружения для инференса содержатся в `requirements_inf.txt`:
```
pip install -r requirements_inf.txt
```
Я создавал окружения через conda. Зависимости окружения для обучения лежат в `requirements_tune.txt`.  
Версия python для запуска экспериментов: 3.10

```
conda create -n qlass_dev_sft python=3.10
conda activate qlass_dev_sft

pip install -r requirements_tune.txt
./setup.sh
apt-get install -y libgl1-mesa-glx
```

### 📑 Data Setup
В зависимости от типа эксперимента требуются разные датасеты. Авторы предоставляют набор "эталонных" траекторий, на которых можно в режиме sft обучить модель. Этот датасет траекторий можно скачать с hugging-face:
```
### download sft json
huggingface-cli download qlass/qlass_sft_data
```

Для воспроизведения экспериментов на Alfworld потребуется также скачать его данные:
```
### download alfworld data
cd eval_agent/data/alfworld
gdown https://drive.google.com/uc?id=1y7Vqeo0_xm9d3I07vZaP6qbPFtyuJ6kI
unzip alfworld_data.zip
```

After setup, the structure of the data folder should look like
```
 
 ├── data/train/
 │   ├── webshop
 │   │   ├── explore                 # Used to store 1-self-explorated output
 │   │   ├── guided_explore          # Used to store 2-1 Q guided exploration output
 │   │   └── webshop_sft.json        # JSON file containing fine-tuning data environment.
 │   ├── alfworld
 │   │   ├── explore
 │   │   ├── guided_explore
 │   │   └── alfworld_sft.json       
 │   └── sciworld
 │       ├── explore
 │       ├── guided_explore
 │       └── sciworld_sft.json       

```
### ⚙️ Resource Requirements
Our scripts are suitable for `4*A6000/A100/H100/A800/H800`. If you want to run on one or two gpus, you can change the logic in the scripts.

Я воспроизводил эксперименты на H200.

### Типы экспериментов
Результаты моих экспериментов содержатся тут: https://docs.google.com/spreadsheets/d/1P9HCtghb5LAUnt0YFwsLh5-GuBTiO2o4Weyucs3VM_4/edit?usp=sharing
#### 1. Классический инференс
Самый наивный эксперимент, который можно воспроизвести - скачать Llama-7B из коробки (`meta-llama/Llama-2-7b-chat-hf`) и воспроизвести инференс этой модели на Alfworld.  
Модель нужно положить в конкретную директорию (например, `~/qlass/models/`) и потом инициализировать переменную `MODEL_PATH` путем до этой модели:
```
export MODEL_PATH=/home/m.iskornev/qlass/models/
```
Далее потребуется использовать скрипт `QLASS/qlass/scripts/eval_sft_7b_alfworld.sh`
В скрипте нужно указать GPU, на которой поднимется sglang сервер с моделью. Перед запуском нужно убедиться, что все параметры в `/home/m.iskornev/qlass/QLASS/qlass/configs/model/sglang_sft.json` выставлены корректно (в частности адрес сервера и температура). Авторы указывают, что для всех экспериментов, кроме сбора датасета для обучения q-сети и кроме q-guided inference, они использовали нулевую температуру, а для указанных двух экспериментов t=0.7.  
Один из важных параметров эксперимента - выбор сплита для тестирования (`--split`). Доступны 2 значения: `dev` - in-distribution сэмпл (или seen) примеров, "похожих" на обучающую выборку, и `test` - out-of-distribution сэмпл (или unseen). Что интересно результаты авторов на `test` были выше чем на `dev`.  
Есди запустить эксперимент с моделью из коробки (без sft дообучения) в качестве финальных метрик должен получиться 0.  
Можно запустить эксперимент с дообученной моделью авторов, веса которой также есть на hugging_face (qlass/qlass-Llama-2-7b-chat-hf-alfworld-sft). В этом случае с нулевой температурой на dev части должно получиться что-то около 66.43, а на test - 70.15.

#### 2. SFT дообучение
Благодаря датасету эталонных траекторий (см. выше) можно дообучить модель в режиме SFT. Для этого потребуется скрипт `sft_7b_alfworld`:
```
MODEL_PATH=/path/to/your/model/ bash ./qlass/scripts/sft_7b_alfworld.sh
```
В этом скрипте важно указать OUT_DIR, куда будут сохраняться итоговые веса модели; и sft_data_path - путь до train данных. При желании обучение можно выполнять на нескольких GPU, но я для экономии ограничился одной. Все остальные параметры обучения в скрипте выставлены в точности, как указано в статье, для получения максимально близких результатов.  
Для обучения важно переключиться на соответствующее окружение.

#### 3. Q-guided inference [1]
Если пользоваться готовой q-net моделью от авторов, веса которой также есть на hugging-face (qlass/qlass-Llama-2-7b-chat-hf-alfworld-Q), то можно сразу перейти к стадии q-guided inference.  
Перед запуском q-guided инференса надо чем-то заполнить OPENAI_API_KEY: `export OPENAI_API_KEY="dummy"`.
Потом как обычно `export MODEL_PATH=/home/m.iskornev/qlass/models/`
И дальше если без nohup:
```
bash ./qlass/scripts/eval_q_wo_perturb_7b_alfworld.sh
```
Этот скрипт поднимет 2 инстанса моделей (sglang сервера - каждый на своей GPU), которые будут сэмплировать возможные действия. На 2-х других GPU поднимаются q-сети, которые оценивают полезность каждого действия. В итоге занимаются 4 GPU (хотя на 2-х остается еще достаточно свободного места). Вместить и sft-модель для генерации траекторий и q-сеть на одну gpu не удалось (и там и там используется Llama на 7B параметров). Чтобы не занимать так много ресурсов я написал скрипт `eval_q_wo_perturb_7b_alfworld_1_gpu.sh`, где инференс ограничивается парой GPU (на одной разворачивается sft-модель, а на другой - q-сеть).  
В скрипте важно корректно указать названия моделей (sft и q-сети) и указать OUT_DIR, куда сохранятся итоговые результаты. В `/QLASS/qlass/configs/model/sglang.json` также настраиваются параметры сервера. Аналогично обычному инференсу можно также управлять сэмплом, на котором тестируется модель с помощью параметра `--split`. Температура в q-guided inference должна быть равна 0.7. Остальные параметры в скрипте выбраны так, чтобы максимально воспроизвести эксперименты авторов.  
  
Я рекомендую запускать эксперимент в nohup, так как инференс на 2-х картах занимает около 8 часов. Для этого следует воспользоваться командами:
```
LOG="logs_q_inf/q_guided_inf_$(date +%F_%H-%M-%S).log"
nohup setsid bash ./qlass/scripts/eval_q_wo_perturb_7b_alfworld.sh </dev/null >"$LOG" 2>&1 & echo $! > logs_q_inf/qnet.pid
```

#### 4. Сбор датасета для обучения q-сети
Можно не пользоваться готовой q-сетью авторов, а обучить свою модель. Но для этого потребуется собрать датасет, в котором для каждого действия указывается его средняя ожидаемая полезность. Таких данных авторы не приводят, поэтому собрать их самостоятельно - единственный вариант. Для этого надо сначала запустить стадию exploration
```
MODEL_PATH=/home/m.iskornev/qlass/models/ bash ./qlass/scripts/explore_7b_alfworld.sh
```
В исходном репозитории авторы поднимают 4 сервера с sft моделью (по одному на каждой GPU) и используя 16 воркеров отправляют на них запросы (каждый воркер занимается своим куском датасета). Все это требуется для построения деревьев траекторий, на основе которых и собирается финальный датасет для обучения q-сети.  
Для экономии ресурсов я поднимал 2 инстанса и обошелся 8 воркерами. Перед запуском важно убедиться, что в директории `/QLASS/qlass/configs/model/` существуют корректные файлы `sglang_explore.json`, в которых указаны параметры серверов. В частности - должны быть указаны валидные порты. Результаты воркеров сохраняются в `--output_dir data/train/${task}/explore_7b_sft_d8_i0_s2_mpr3/`.  
Скрипт explore_7b_alfworld.sh работает продолжительное время, поэтому его лучше тоже запускать в фоне, например, через nohup.
  
После завершения работы explore_7b_alfworld.sh важно собрать полученные данные в датасет, подходящий для обучения модели. Для этого требуется следующий скрипт - `collect_q.sh`, в котором надо указать директорию, куда писал данные explore_7b_alfworld.sh. Итоговые данные будут лежать в `data/train/${task}/explore/`. В collect_q.sh есть параметр `--q_type`, отвечающий за алгоритм дисконтирования награды от терминальных вершин. Классичесая реализация qlass предполагает значения vanilla.  
  
На всякий случай я залил мой датасет (которые я собрал самостоятельно) на google drive: https://drive.google.com/file/d/1e2E27EqfmkWdJ5JcmLgSO4JmFJJX6kVF/view?usp=sharing

#### 5. Обучение q-сети
После сбора данных можно запускать обучение q-сети. Этот процесс запускается с помощью скрипта `train_qnet_7b.sh`. В исходном репозитории обучение производилось на 4-х картах, но я уменьшил до 2-х для экономии ресурсов. Результаты обучения сохраняются в `${MODEL_PATH}/${q_model_name}`. Для обучения нужно также использовать соответствующее окружение (зависимости из `requirements_tune.txt`).  
Рекомендую тоже запускать через nohup. Для удобства я написал еще один скрипт-обертку `/QLASS/run_qnet_nohup/sh`. Команды для запуска:
```
LOG="logs/qnet_$(date +%F_%H-%M-%S).log"
nohup setsid ./run_qnet_nohup.sh </dev/null >"$LOG" 2>&1 & echo $! > logs/qnet.pid
```
Данные для обучения скрипт забирает из `data/train/${task}/explore/vanilla.jsonl` (куда их должен был положить collect_q.sh).  
Само обучение происходит достаточно долго (из-за того что обучается вся сеть). На 2-х H200 обучение длилось примерно 3-4 дня.

#### 6. Q-guided inference [2]
После обучения q-сети можно запускать q-guided инференс с этой моделью (чтобы они оценивала действия). Для этого достаточно указать соответствующее название для q_model_name. Однако здесь есть небольшой нюанс. После обучения в директории `/home/m.iskornev/qlass/models/qlass-Llama-2-7b-chat-hf-alfworld-Q_my` у меня появилась директория с чекпоинтом `checkpoint-14382/`, однако запустить инференс с этим чекпоинтом напрямую не удавалось. Пришлось провести следующие манипуляции:
 - создать директорию infer-checkpoint-14382
 - положить в нее pytorch_model_fsdp.bin из checkpoint-14382 и переименовать его в pytorch_model.bin
 - положить файлы tokenizer.model, tokenizer_config.json, special_tokens_map.json из checkpoint-14382
 - положить config.json из родительской директории `/models/qlass-Llama-2-7b-chat-hf-alfworld-Q_my`
  
В итоге после указания в eval_q_wo_perturb_7b_alfworld.sh пути до infer-checkpoint-14382, удалось запустить q-guided инференс.



  
---
Важный нюанс: все эксперименты воспроизводиилсь на бенчмарке Alfworld. В статье также рассматриваюся Sciworld и Webshop. Но относительно Webshop авторы отмечают, что там много однотипных задач и как следствие однотипных действий, которые выбирает агент. Чтобы разнообразить действия авторы переформулируют задачи с помощью GPT-3.5.
  
  

---
На всякий случай инструкция для запуска экспериментов от авторов, но она менее подробная:  
  


### 🚀 Run the Q-guided inference
We show how to directly use the well-trained QNet to run inference.
First Download SFT model from https://huggingface.co/qlass/qlass-Llama-2-7b-chat-hf-alfworld-sft and Q-Net from https://huggingface.co/qlass/qlass-Llama-2-7b-chat-hf-alfworld-Q and put them in `MODEL_PATH`
Before you start, make sure you have correct `sglang*.json` in `configs/agent/model`
```
bash ./qlass/scripts/eval_q_wo_perturb_7b_alfworld.sh ## we use no perturbation version for alfworld, this step will generate eval results files in {output_dir}

python ./qlass/calc_results.py ## collect the final results (you can change the path according to {output_dir} inside the code)
```

### 🎮 Run the whole pipeline
Download Llama-2-7b-chat-hf from https://huggingface.co/meta-llama/Llama-2-7b-chat-hf and put it in `MODEL_PATH`

Before you start, make sure you have correct `sglang*.json` in `configs/agent/model`

```
### SFT the model
MODEL_PATH=/path/to/your/model bash ./qlass/scripts/sft_7b_alfworld.sh

### Eval sft model (you can change split to test on either dev/test set)
MODEL_PATH=/path/to/your/model bash ./qlass/scripts/eval_sft_7b_alfworld.sh

### Exploration
MODEL_PATH=/path/to/your/model bash ./qlass/scripts/explore_7b_alfworld.sh

### Collect Q Data
bash ./qlass/scripts/collect_q.sh

### Train Q-Net
bash ./qlass/scripts/train_qnet_7b.sh

### Q-guided inference
bash ./qlass/scripts/eval_q_wo_perturb_7b_alfworld.sh ## we use no perturbation version for alfworld, this step will generate eval results files in {output_dir}

python ./qlass/calc_results.py ## collect the final results (you can change the path according to {output_dir} inside the code)

### if you want to use q-inference with perturbation
export OPENAI_ORG={YOUR_ORG}
export OPENAI_API_KEY={YOUR_KEY}
bash ./qlass/scripts/eval_q_perturb_7b_alfworld.sh ## we use no perturbation version for alfworld


```

### 🔧 Some Common Issues & solutions
```
BUG: libstdc++.so.6: version `GLIBCXX_3.4.29' not found
SOLUTION: https://github.com/pybind/pybind11/discussions/3453

BUG: Exception: Unable to find javac
SOLUTION: https://stackoverflow.com/questions/5736641/ant-unable-to-find-javac-java-home-wont-set-on-ubuntu/37201765#37201765

```

### 🌹 Acknowledgement
We borrowed some implementations from https://github.com/Yifan-Song793/ETO and https://github.com/sgl-project/sglang. Thanks for their great work!

### 📖 Citation

If you find this repo helpful, please cite out paper:

```
@article{lin2025qlass,
  title={QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search},
  author={Lin, Zongyu and Tang, Yao and Yao, Xingcheng and Yin, Da and Hu, Ziniu and Sun, Yizhou and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:2502.02584},
  year={2025}
}
```
