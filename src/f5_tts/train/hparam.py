from f5_tts.model.modules import LoraConfig, ConvAdapterConfig

adapt_size = 0.25 # 0.25, 1, 2, 4
kernel_size = 3 # 7, 3
blocks = [0,1,2,3] 
prompt_rank = 64 # 8, 16, 32, 64
dit_rank = 32 # 8, 16, 32, 64   -> N12 : 64, N13 : 32  N15 :64
drop_path = 0.1 # 0.1, 0.3, 0.5, 0.7, 0.9

adpt_dict = dict(text_embed="adapter", input_embed="adapter", transformer_blocks="adapter") # "full", "freeze", "adpater"

#base_config = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
# Conditioning Adapter
#conditioning_adapter_config = {'method' : 'conv_adapt', 'adapt_size':adapt_size, 'adapt_scale': 1.0, 'kernel_sioze': kernel_size, 'blocks': blocks}
conditioning_adapter_config = ConvAdapterConfig(method= 'conv_adapt', adapt_size= adapt_size, kernel_size= kernel_size) # if use conditioning adapter

# Prompt Adapter
prompt_adapter_config = LoraConfig(r=prompt_rank, lora_alpha = 2 * prompt_rank, target_modules=["proj"], scale=1.0, drop_path = drop_path)


# DiT LoRA Adapter
dit_lora_adapter_config = LoraConfig(r=dit_rank, lora_alpha =2 * dit_rank, target_modules = ["to_q","to_v", "to_v"], scale=1.0)#, lora_adapter_name="randlora") # lora


def is_adapter(config):
    if adpt_dict[config] == "adapter" :
        return True
    else :
        return False

def get_model_cfg(base_config):
    model_configs = {
        "conditioning_adapter_config" : conditioning_adapter_config if is_adapter("text_embed") else None,
        "prompt_adapter_config" : prompt_adapter_config if is_adapter("input_embed") else None,
        "dit_lora_adapter_config" : dit_lora_adapter_config if is_adapter("transformer_blocks") else None,
        "ko": False,
    }
    return({**base_config, **model_configs} , {**model_configs}, {**adpt_dict})