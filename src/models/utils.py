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
                spatial_dims=args.spatial_dims,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                channels=args.channels,
                strides=args.strides,
                kernel_size=args.kernel_size,
                up_kernel_size=args.up_kernel_size,
                num_res_units=args.num_res_units,
                dropout=args.dropout,
                bias=args.bias,
                adn_ordering=args.adn_ordering
            )
        
    else:
        raise ValueError("")