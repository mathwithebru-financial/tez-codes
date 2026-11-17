"""Temporary numerical adapter for six fixed P1 V2 models; no imports of torch.

The saved CUDA encoder fastpath uses a GELU epilogue. A separately verified
unfused tanh-GELU implementation permits input derivatives. This context changes
ONLY the four encoder activation callables, never the two head GELU modules.
Finite forward agreement is not a global function/gradient identity proof.
"""
import contextlib

ADAPTER_ID='temporary_encoder_only_tanh_GELU_MHAoff_SDPAmath_v2'


def require(condition,message):
    if not condition:raise ValueError(message)


def encoder_layers(model):
    layers=[]
    for branch in (model.ret_encoder,model.vol_encoder):
        part=list(branch.encoder.layers)
        require(len(part)==2,'The selected model must have two encoder layers per branch')
        layers.extend(part)
    require(len(layers)==4 and len({id(x) for x in layers})==4,'Exactly four distinct encoder layers required')
    return layers


@contextlib.contextmanager
def compatible_encoder_input_gradients(model,torch):
    require(model.training is False,'Explanation adapter requires eval mode')
    parameters=list(model.parameters())
    require(parameters and all(not p.requires_grad and p.grad is None for p in parameters),'All model parameter gradients must be frozen and empty')
    layers=encoder_layers(model)
    original_activations=[layer.activation for layer in layers]
    require(all(callable(a) for a in original_activations),'Invalid encoder activation')
    heads=[(name,module,module.approximate) for name,module in model.named_modules() if isinstance(module,torch.nn.GELU)]
    require(len(heads)==2 and all(name.startswith(('return_head.','vol_head.')) and approximation=='none' for name,module,approximation in heads),'Expected two unchanged exact-GELU head modules')
    previous_fastpath=torch.backends.mha.get_fastpath_enabled()
    def encoder_tanh_gelu(values):
        return torch.nn.functional.gelu(values,approximate='tanh')
    try:
        for layer in layers:layer.activation=encoder_tanh_gelu
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            yield {'adapter_id':ADAPTER_ID,'encoder_activation_count':4,
                'head_GELU_approximation_unchanged':'none','head_count':2,
                'finite_forward_agreement_is_not_global_gradient_identity':True}
    finally:
        for layer,activation in zip(layers,original_activations):layer.activation=activation
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
        require(all(layer.activation is activation for layer,activation in zip(layers,original_activations)),'Original encoder activation objects were not restored')
        require(all(module.approximate==approximation for name,module,approximation in heads),'A head GELU module changed')
        current_heads=[(name,module) for name,module in model.named_modules() if isinstance(module,torch.nn.GELU)]
        require(len(current_heads)==2 and all(a[0]==b[0] and a[1] is b[1] for a,b in zip(current_heads,heads)),'Head module identities changed')
        require(all(not p.requires_grad and p.grad is None for p in parameters),'Adapter produced parameter gradients')
