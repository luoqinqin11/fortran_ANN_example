PROGRAM main_neural_example
    USE neural
    IMPLICIT NONE
    ! --- 在主程序中定义 dp 参数，使其在主程序作用域内可见 ---
    INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(15, 307)

    TYPE(network) :: my_net
    INTEGER, DIMENSION(3) :: layer_sizes = (/2, 4, 1/) ! 输入层2个神经元，隐藏层4个，输出层1个

    DOUBLE PRECISION, DIMENSION(4, 2) :: train_inputs
    DOUBLE PRECISION, DIMENSION(4, 1) :: train_outputs

    DOUBLE PRECISION, DIMENSION(2) :: test_input
    DOUBLE PRECISION, DIMENSION(1) :: prediction_output

    CHARACTER(LEN=30) :: network_save_file = "xor_network1.nn"

    ! --- 1. 定义训练数据 (XOR 问题) ---
    train_inputs = RESHAPE((/0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, 1.0_dp/), SHAPE(train_inputs))
    train_outputs = RESHAPE((/0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp/), SHAPE(train_outputs))

    PRINT *, "--- Initializing Network ---"
    CALL neural_init_network(my_net, layer_sizes)
    PRINT *, "Network layers:", my_net%nl
    PRINT *, "Layer sizes:", my_net%ls

    PRINT *, "--- Training Network ---"
    ! 训练10000次，学习率为0.1
    CALL neural_train_network_direct(my_net, train_inputs, train_outputs, 10000, 0.1_dp) ! <-- 这里修改
    PRINT *, "Training complete."

    PRINT *, "--- Testing Network Predictions ---"
    test_input = (/0.0_dp, 0.0_dp/)
    prediction_output = neural_use_network(my_net, test_input)
    PRINT *, "Input (0,0) -> Output:", prediction_output

    test_input = (/0.0_dp, 1.0_dp/)
    prediction_output = neural_use_network(my_net, test_input)
    PRINT *, "Input (0,1) -> Output:", prediction_output

    test_input = (/1.0_dp, 0.0_dp/)
    prediction_output = neural_use_network(my_net, test_input)
    PRINT *, "Input (1,0) -> Output:", prediction_output

    test_input = (/1.0_dp, 1.0_dp/)
    prediction_output = neural_use_network(my_net, test_input)
    PRINT *, "Input (1,1) -> Output:", prediction_output

    PRINT *, "--- Saving Network ---"
    CALL neural_save_network(my_net, network_save_file)
    PRINT *, "Network saved to:", network_save_file

    ! --- 重新加载网络以验证保存功能 ---
    PRINT *, "--- Loading Network ---"
    CALL neural_load_network(my_net, network_save_file)
    PRINT *, "Network loaded from:", network_save_file

    PRINT *, "--- Testing Loaded Network ---"
    test_input = (/0.0_dp, 1.0_dp/)
    prediction_output = neural_use_network(my_net, test_input)
    PRINT *, "Input (0,1) -> Output (from loaded network):", prediction_output

END PROGRAM main_neural_example