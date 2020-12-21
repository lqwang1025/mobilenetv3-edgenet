import geffnet
# models can also be created by using the entrypoint directly
def main()
   m = geffnet.efficientnet_b0(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2)
   m.train()

