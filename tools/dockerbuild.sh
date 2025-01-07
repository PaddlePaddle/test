function docker_build(){
    local PRODUCT_NAME=$1
    local Dokckerfile=$2
    # 使用--no-cache参数进行docker build操作
    docker build --progress=plain -t ${PRODUCT_NAME} -f ${Dokckerfile} . \
        --network host \
        --no-cache \
        --build-arg HTTP_PROXY=${http_proxy} \
        --build-arg HTTPS_PROXY=${http_proxy} \
        --build-arg NO_PROXY=${no_proxy} \
        --build-arg ftp_proxy=${http_proxy}
}

function print_usage() {
    echo "Usage: $0 {command}"
    echo "Commands:"
    echo "  docker_build_paddle_model_test    Build Docker image for Paddle Model Test"
    echo "  help                              Display this usage information"
}


function main() {
    local CMD=$1
    case $CMD in
        docker_build_paddle_model_test)
        PRODUCT_NAME='registry.baidubce.com/paddlepaddle/paddleqa:Paddle-Model-Test-v1.0.0'
        Dokckerfile='./dockerfiles/Dockerfile.Paddle-Model-Test-v1.0.0'
        docker_build  ${PRODUCT_NAME} ${Dokckerfile}
        ;;
        help)
        print_usage
        ;;
        *)
        print_usage
        exit 1
        ;;
    esac
}

main $@
