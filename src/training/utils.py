def get_dataset(args):

    if args.dimension == '2d':
        if args.dataset == "amos22":
            from .dim2.dataset_amos22 import DatasetAmos22
            return DatasetAmos22(args=args)
        
    elif args.dimension == '3d':
        if args.dataset == "amos22":
            from .dim3.dataset_amos22 import DatasetAmos22
            return DatasetAmos22(args=args)
        
    else:
        raise ValueError("")
