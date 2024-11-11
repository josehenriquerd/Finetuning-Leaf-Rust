import torch
import logging
from transformers import AutoTokenizer, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig, LlavaNextProcessor
from trl import SFTTrainer
from datasets import load_dataset
from peft import PeftModel, LoraConfig, TaskType

# Definir o ID do modelo
model_id = "llava-hf/llava-1.5-7b-hf"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verificar se a GPU está disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Configurações de quantização
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Carregar o modelo base e configurá-lo para quantização
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)

# Carregar e configurar o processador
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model.config.patch_size = processor.patch_size
model.config.vision_feature_select_strategy = processor.vision_feature_select_strategy
model.processor = processor  # Associar o processador diretamente ao modelo

# Configurações de LoRA para adicionar camadas treináveis ao modelo quantizado
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,  # Dimensão do rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
# Aplicar LoRA ao modelo base usando a configuração especificada
model = PeftModel(model, peft_config).to(device)

# Carregar o tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Template de chat para o tokenizer
LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions. \
All questions and images pertain to leaves affected by rust. \
{% for message in messages %}{% if message['role'] == 'user' %} \
USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %} \
{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""

tokenizer.chat_template = LLAVA_CHAT_TEMPLATE

# Carregar o dataset e definir o mapeamento das classes
dataset_dir = "C:/Users/NPC TECH/Desktop/tcc/dataset_local"
raw_datasets = load_dataset("imagefolder", data_dir=dataset_dir)
train_dataset = raw_datasets['train']
eval_dataset = raw_datasets['test']

# Função para adicionar mensagens e pixel_values aos exemplos do dataset
def add_message_and_pixel_values(batch):
    messages, texts, labels, pixel_values = [], [], [], []
    
    for i, example in enumerate(batch['image']):
        messages.append([{'role': 'user', 'content': [{'text': 'folha com ferrugem', 'type': 'text'}]}])
        texts.append('folha com ferrugem')
        labels.append(1)  # Classe para 'folha com ferrugem'

        try:
            # Processar a imagem
            inputs = processor(images=example, text='folha com ferrugem', return_tensors="pt")
            
            # Verificar se 'pixel_values' foi gerado corretamente
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'][0])  # Assumindo tensor
                print(f"Exemplo {i}: 'pixel_values' gerado com sucesso.")
            else:
                print(f"Aviso: 'pixel_values' não encontrado para o exemplo {i}.")
                
        except Exception as e:
            print(f"Erro ao processar a imagem para o exemplo {i}: {e}")
    
    return {
        'messages': messages,
        'text': texts,
        'label': labels,
        'pixel_values': pixel_values
    }

# Aplicar a função ao dataset em batch
train_dataset = train_dataset.map(add_message_and_pixel_values, batched=True, batch_size=1)  # Ajuste o batch_size conforme necessário
eval_dataset = eval_dataset.map(add_message_and_pixel_values, batched=True, batch_size=1)

print("Visualização do dataset após o mapeamento:")
print(train_dataset)


# Função de collate personalizada para garantir que pixel_values seja passado corretamente
# Função de collate personalizada com debug adicional
def custom_collate_fn(batch):
    # Filtrar exemplos que possuem 'pixel_values' válidos
    batch = [example for example in batch if 'pixel_values' in example and example['pixel_values'] is not None]
    
    # Imprima o conteúdo do batch para depuração
    if not batch:
        print("Erro: Batch vazio após filtrar exemplos válidos.")
        raise ValueError("Batch vazio: nenhum exemplo válido com 'pixel_values' encontrado.")
    else:
        print("Batch atual:", batch)
    
    # Criar o tensor com 'pixel_values'
    pixel_values = torch.stack([example['pixel_values'] for example in batch])
    labels = torch.tensor([example['label'] for example in batch])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

# Debug adicional para o dataset após o mapeamento
print("Verificação final do dataset antes do treinamento:")
#print("Exemplo do dataset com pixel_values:", train_dataset[0] if 'pixel_values' in train_dataset[0] else "pixel_values ausente")

# Configuração do SFTTrainer
# Configurações de treinamento
training_args = TrainingArguments(
    output_dir="./llava-output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    eval_strategy="steps",
    logging_dir="./logs",
    logging_steps=500,
    load_best_model_at_end=True,
)

# Instanciar o trainer com a função de collate personalizada
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=custom_collate_fn
)

# Iniciar o treinamento com configuração ajustada
print("Iniciando o treinamento com configuração ajustada.")
trainer.train()