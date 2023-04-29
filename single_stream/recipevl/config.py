import argparse

def get_args():

    parser = argparse.ArgumentParser()
    # arguments for dataset
    parser.add_argument("--data_dir", default='/data/s2478846/data', type=str, required=False,
                        help="The input data dir with all required files.")
    # parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--im_size", default=224, type=int)
    parser.add_argument("--max_seq_len", default=350, type=int)
    
    # settings for model
    parser.add_argument("--load_path",  type=str, default='')
    parser.add_argument("--num_workers", type=int, default=16)
    
    # parser.add_argument("--max_image_len", type=int, default=200)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--whole_word_masking", type=bool, default=True)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--vit", type=str, default= "vit_small_patch16_224")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    
    #training 
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast_run", action='store_true', help="Whether to run fast.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help='node rank for distributed training')
    
    parser.add_argument("--resume_from", type=str, default='')
    parser.add_argument("--output_dir", default='output/', type=str)
    parser.add_argument("--model_name", default='', type=str)
    
    parser.add_argument("--extract_info", action='store_true')
    
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=10)
    parser.add_argument("--evaluate_during_training", action='store_true', 
                        help="Run evaluation during training at each epoch.")      
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--eval_sample", type=int, default=500)
    parser.add_argument("--eval_times", type=int, default=1)
    
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0.1, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler_name", default='linear', type=str, help="learning rate scheduler.")
    parser.add_argument("--logging_steps", type = int, default=20)
    
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation.")
    
    
                    
    args = parser.parse_args()
    return args                    

    

