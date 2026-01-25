# SI: Scaled Iters, by number of neuron groups
# python -m tasks.image_classification.train_ctm2 \
# --log_dir logs/cifar100/ctm2/d=512x1--i=16--heads=4--sd=2--synch=256-256-0-h=16-random-pairing--iters=25K--backbone=18-2--seed=42 \
# --model ctm2 \
# --dataset cifar100 \
# --d_model 512 \
# --n_neuron_groups 1 --group_router_type mlp --use_tick_conditioned_routing --dropout_router 0.15 \
# --d_input 256 \
# --synapse_depth 2 \
# --heads 4 \
# --n_synch_out 256 \
# --n_synch_action 256 \
# --n_random_pairing_self 0 \
# --neuron_select_type random-pairing \
# --iterations 16 \
# --memory_length 25 \
# --deep_memory \
# --memory_hidden_dims 16 \
# --dropout 0.0 \
# --dropout_nlm 0 \
# --do_normalisation \
# --positional_embedding_type none \
# --backbone_type resnet18-2 \
# --training_iterations 25000 \
# --warmup_steps 1000 \
# --use_scheduler \
# --scheduler_type cosine \
# --weight_decay 0.001 \
# --save_every 1000 \
# --track_every 2000 \
# --n_test_batches -1 \
# --num_workers_train 8 \
# --batch_size 1024 \
# --batch_size_test 1024 \
# --lr 1e-4 \
# --device 0 \
# --seed 42
# same as 15jan model

python -m tasks.image_classification.train_ctm2 \
--log_dir logs/cifar100/ctm2/SI--d=512x2--i=16--heads=4--sd=2--synch=256-256-0-h=16-random-pairing--iters=25K--backbone=18-2--seed=42 \
--model ctm2 \
--dataset cifar100 \
--d_model 512 \
--n_neuron_groups 2 --group_router_type mlp --use_tick_conditioned_routing --dropout_router 0.15 \
--d_input 256 \
--synapse_depth 2 \
--heads 4 \
--n_synch_out 256 \
--n_synch_action 256 \
--n_random_pairing_self 0 \
--neuron_select_type random-pairing \
--iterations 16 \
--memory_length 25 \
--deep_memory \
--memory_hidden_dims 16 \
--dropout 0.0 \
--dropout_nlm 0 \
--do_normalisation \
--positional_embedding_type none \
--backbone_type resnet18-2 \
--training_iterations 50000 \
--warmup_steps 1000 \
--use_scheduler \
--scheduler_type cosine \
--weight_decay 0.001 \
--save_every 1000 \
--track_every 2000 \
--n_test_batches -1 \
--num_workers_train 8 \
--batch_size 1024 \
--batch_size_test 1024 \
--lr 1e-4 \
--device 0 \
--seed 42

python -m tasks.image_classification.train_ctm2 \
--log_dir logs/cifar100/ctm2/SI--d=512x4--i=16--heads=4--sd=2--synch=256-256-0-h=16-random-pairing--iters=25K--backbone=18-2--seed=42 \
--model ctm2 \
--dataset cifar100 \
--d_model 512 \
--n_neuron_groups 4 --group_router_type mlp --use_tick_conditioned_routing --dropout_router 0.15 \
--d_input 256 \
--synapse_depth 2 \
--heads 4 \
--n_synch_out 256 \
--n_synch_action 256 \
--n_random_pairing_self 0 \
--neuron_select_type random-pairing \
--iterations 16 \
--memory_length 25 \
--deep_memory \
--memory_hidden_dims 16 \
--dropout 0.0 \
--dropout_nlm 0 \
--do_normalisation \
--positional_embedding_type none \
--backbone_type resnet18-2 \
--training_iterations 100000 \
--warmup_steps 1000 \
--use_scheduler \
--scheduler_type cosine \
--weight_decay 0.001 \
--save_every 1000 \
--track_every 2000 \
--n_test_batches -1 \
--num_workers_train 8 \
--batch_size 1024 \
--batch_size_test 1024 \
--lr 1e-4 \
--device 0 \
--seed 42

python -m tasks.image_classification.train_ctm2 \
--log_dir logs/cifar100/ctm2/SI--d=512x8--i=16--heads=4--sd=2--synch=256-256-0-h=16-random-pairing--iters=25K--backbone=18-2--seed=42 \
--model ctm2 \
--dataset cifar100 \
--d_model 512 \
--n_neuron_groups 8 --group_router_type mlp --use_tick_conditioned_routing --dropout_router 0.15 \
--d_input 256 \
--synapse_depth 2 \
--heads 4 \
--n_synch_out 256 \
--n_synch_action 256 \
--n_random_pairing_self 0 \
--neuron_select_type random-pairing \
--iterations 16 \
--memory_length 25 \
--deep_memory \
--memory_hidden_dims 16 \
--dropout 0.0 \
--dropout_nlm 0 \
--do_normalisation \
--positional_embedding_type none \
--backbone_type resnet18-2 \
--training_iterations 200000 \
--warmup_steps 1000 \
--use_scheduler \
--scheduler_type cosine \
--weight_decay 0.001 \
--save_every 1000 \
--track_every 2000 \
--n_test_batches -1 \
--num_workers_train 8 \
--batch_size 1024 \
--batch_size_test 1024 \
--lr 1e-4 \
--device 0 \
--seed 42


# python -m tasks.image_classification.train_ctm2 \
# --log_dir logs/cifar100/ctm2/d=1024x1--i=16--heads=4--sd=2--synch=256-256-0-h=16-random-pairing--iters=25K--backbone=18-2--seed=42 \
# --model ctm2 \
# --dataset cifar100 \
# --d_model 1024 \
# --n_neuron_groups 1 --group_router_type mlp --use_tick_conditioned_routing --dropout_router 0.15 \
# --d_input 256 \
# --synapse_depth 2 \
# --heads 4 \
# --n_synch_out 256 \
# --n_synch_action 256 \
# --n_random_pairing_self 0 \
# --neuron_select_type random-pairing \
# --iterations 16 \
# --memory_length 25 \
# --deep_memory \
# --memory_hidden_dims 16 \
# --dropout 0.0 \
# --dropout_nlm 0 \
# --do_normalisation \
# --positional_embedding_type none \
# --backbone_type resnet18-2 \
# --training_iterations 25000 \
# --warmup_steps 1000 \
# --use_scheduler \
# --scheduler_type cosine \
# --weight_decay 0.001 \
# --save_every 1000 \
# --track_every 2000 \
# --n_test_batches -1 \
# --num_workers_train 8 \
# --batch_size 1024 \
# --batch_size_test 1024 \
# --lr 1e-4 \
# --device 0 \
# --seed 42

python -m tasks.image_classification.train_ctm2 \
--log_dir logs/cifar100/ctm2/SI--d=1024x2--i=16--heads=4--sd=2--synch=256-256-0-h=16-random-pairing--iters=25K--backbone=18-2--seed=42 \
--model ctm2 \
--dataset cifar100 \
--d_model 1024 \
--n_neuron_groups 2 --group_router_type mlp --use_tick_conditioned_routing --dropout_router 0.15 \
--d_input 256 \
--synapse_depth 2 \
--heads 4 \
--n_synch_out 256 \
--n_synch_action 256 \
--n_random_pairing_self 0 \
--neuron_select_type random-pairing \
--iterations 16 \
--memory_length 25 \
--deep_memory \
--memory_hidden_dims 16 \
--dropout 0.0 \
--dropout_nlm 0 \
--do_normalisation \
--positional_embedding_type none \
--backbone_type resnet18-2 \
--training_iterations 50000 \
--warmup_steps 1000 \
--use_scheduler \
--scheduler_type cosine \
--weight_decay 0.001 \
--save_every 1000 \
--track_every 2000 \
--n_test_batches -1 \
--num_workers_train 8 \
--batch_size 1024 \
--batch_size_test 1024 \
--lr 1e-4 \
--device 0 \
--seed 42

python -m tasks.image_classification.train_ctm2 \
--log_dir logs/cifar100/ctm2/SI--d=1024x4--i=16--heads=4--sd=2--synch=256-256-0-h=16-random-pairing--iters=25K--backbone=18-2--seed=42 \
--model ctm2 \
--dataset cifar100 \
--d_model 1024 \
--n_neuron_groups 4 --group_router_type mlp --use_tick_conditioned_routing --dropout_router 0.15 \
--d_input 256 \
--synapse_depth 2 \
--heads 4 \
--n_synch_out 256 \
--n_synch_action 256 \
--n_random_pairing_self 0 \
--neuron_select_type random-pairing \
--iterations 16 \
--memory_length 25 \
--deep_memory \
--memory_hidden_dims 16 \
--dropout 0.0 \
--dropout_nlm 0 \
--do_normalisation \
--positional_embedding_type none \
--backbone_type resnet18-2 \
--training_iterations 100000 \
--warmup_steps 1000 \
--use_scheduler \
--scheduler_type cosine \
--weight_decay 0.001 \
--save_every 1000 \
--track_every 2000 \
--n_test_batches -1 \
--num_workers_train 8 \
--batch_size 1024 \
--batch_size_test 1024 \
--lr 1e-4 \
--device 0 \
--seed 42

python -m tasks.image_classification.train_ctm2 \
--log_dir logs/cifar100/ctm2/SI--d=1024x8--i=16--heads=4--sd=2--synch=256-256-0-h=16-random-pairing--iters=25K--backbone=18-2--seed=42 \
--model ctm2 \
--dataset cifar100 \
--d_model 1024 \
--n_neuron_groups 8 --group_router_type mlp --use_tick_conditioned_routing --dropout_router 0.15 \
--d_input 256 \
--synapse_depth 2 \
--heads 4 \
--n_synch_out 256 \
--n_synch_action 256 \
--n_random_pairing_self 0 \
--neuron_select_type random-pairing \
--iterations 16 \
--memory_length 25 \
--deep_memory \
--memory_hidden_dims 16 \
--dropout 0.0 \
--dropout_nlm 0 \
--do_normalisation \
--positional_embedding_type none \
--backbone_type resnet18-2 \
--training_iterations 200000 \
--warmup_steps 1000 \
--use_scheduler \
--scheduler_type cosine \
--weight_decay 0.001 \
--save_every 1000 \
--track_every 2000 \
--n_test_batches -1 \
--num_workers_train 8 \
--batch_size 1024 \
--batch_size_test 1024 \
--lr 1e-4 \
--device 0 \
--seed 42



