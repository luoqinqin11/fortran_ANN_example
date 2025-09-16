program run_xor_minimal
    use neural
    implicit none

    type(network) :: testNet
    real(8), dimension(2) :: input
    real(8), dimension(:), allocatable :: output

    ! 初始化网络，结构 2 -> 3 -> 3 -> 1
    call neural_init_network(testNet, (/ 2, 3, 3, 1 /))
    call neural_train_network_file(testNet, "xor.dat", 10000)
    ! --- 不进行训练，直接用随机初始化的网络测试 ---
    
    ! 测试 1 1
    input = (/ 1.0d0, 1.0d0 /)
    output = neural_use_network(testNet, input)
    print *, "Input: ", input, " Output: ", output

    ! 测试 0 1
    input = (/ 0.0d0, 1.0d0 /)
    output = neural_use_network(testNet, input)
    print *, "Input: ", input, " Output: ", output

    ! 测试 1 0
    input = (/ 1.0d0, 0.0d0 /)
    output = neural_use_network(testNet, input)
    print *, "Input: ", input, " Output: ", output

    ! 测试 0 0
    input = (/ 0.0d0, 0.0d0 /)
    output = neural_use_network(testNet, input)
    print *, "Input: ", input, " Output: ", output

end program run_xor_minimal

    
