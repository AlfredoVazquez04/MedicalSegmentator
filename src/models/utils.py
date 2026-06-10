def get_model(args):
    if args.dimension == '2d':
        if args.model == 'unet':
            from .dim2 import UNet
            return UNet(

            )
        
    elif args.dimension == '3d':
        if args.model == 'unet':
            from .dim3 import UNet
            return UNet(

            )
        
    else:
        raise ValueError("")