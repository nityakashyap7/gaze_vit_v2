class AttentionExtractor:
    def __init__(self, model):
        # might want to regularize multiple of the last layers, maybe turn this into some kinda for loop and in my config file i can specify how many of the last layers i wana regularize and with what intensity?
        spatial_transformer_last_layer = model.spatial_transformer.layers[-1][0].attend
        # The handle below is just a receipt (u dont need to store it bc its only job is to let you remove the hook later if you want to, via self.handle.remove())
        self.handle = spatial_transformer_last_layer.register_forward_hook(self.hook_fn) # type: ignore 
        spatial_transformer_last_layer.use_flash_attn = False # disable flash attention for the same layer(s) ur extracting weights from 
        

        self.cls_qkt_logits = None 

    def remove(self):
        self.handle.remove()
    
    def hook_fn(self, layer, input, output):
        # im leaving out the processing logic and keeping the hook dumb- softmaxing is part of the regularization computation, the hook is only supposed to return what it gets from the model
        attn_weights = input[0] # input comes in as a tuple wrapping the input tensor so u need to index into it

        # attn_weights shape: [batch_size, num_heads, num_patches, num_patches]. here num_patches includes CLS token.
        self.cls_qkt_logits = attn_weights[:, :, 0, 1:] # leave 1st and 2 dims alone, grab only 1st row of third (cls dot prods), exclude 1st element of 4th dim (bc that's cls self attn)