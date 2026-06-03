#!/bin/bash
set -e

# 检查 AWS CLI 是否安装
if ! command -v aws &> /dev/null; then
    echo "AWS CLI 未安装，开始安装..."
    
    # 更新软件包列表并安装 unzip 与 zip
    apt-get update
    apt-get install -y unzip zip
    
    # 下载 AWS CLI 压缩包
    wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -O "awscliv2.zip"
    
    # 解压
    unzip awscliv2.zip
    
    # 安装 AWS CLI
    ./aws/install

    # 清理安装文件
    rm -rf aws awscliv2.zip
else
    echo "AWS CLI 已安装."
fi

# 创建 ~/.aws 目录（若不存在）
mkdir -p ~/.aws

# 写入 ~/.aws/credentials 文件
cat > ~/.aws/credentials <<'EOF'
[default]
aws_access_key_id = XsoNq0QPQJToqHsG
aws_secret_access_key = ylQHuuY8LIkIbjSHqsoW0tUD9aehQQse
[cv_data]
aws_access_key_id = 57d71c51a2dae1ea
aws_secret_access_key = aeef4e8cce5a6e8b55548b4788741bd3
[cv_platform]
aws_access_key_id = f0e7b8267d8d4bc7
aws_secret_access_key = 4735f36a08ce53d8d87767b96472d075
[llm_snapshot]
aws_access_key_id = T90UlXgHTqGfovOh
aws_secret_access_key = tPIA6vHV7Jxkpmr4Ria75pj773AqDAZI
[llm-data]
aws_access_key_id = XsoNq0QPQJToqHsG
aws_secret_access_key = ylQHuuY8LIkIbjSHqsoW0tUD9aehQQse
[ai_models]
aws_access_key_id = d92cc9a495da7edc
aws_secret_access_key = 107108c8e45e09fa6521b99f04fb4100
[ai_llm_qa]
aws_access_key_id = Zxw3Z1g2RMNOz9cY
aws_secret_access_key = NZhMConbiqJ4ypUeNnyq3lFnoZFi5kIy
[ai_llm_models]
aws_access_key_id = y6h9ryjKPo8vyyqh
aws_secret_access_key = qtn2dX3r8K3McZfVMQjwWJeSHrGCDFn5
[llm_models_70b]
aws_access_key_id = kwwtoKJ6RwZCKQ3s
aws_secret_access_key = eAUh08q2UmekXvzYuUs7u3TOLmrTBEtf
[reply_model]
aws_access_key_id = 9SpmlZEtHEsaMrja
aws_secret_access_key = ZcqKzvjWf0OMvbzQnyxsia3kCxKKhRb0
EOF

# 写入 ~/.aws/config 文件
cat > ~/.aws/config <<'EOF'
[default]
region = shjd-inner
[profile cv_data]
s3=
    max_concurrent_requests = 5
region = jssz
[profile cv_platform]
region = jssz
[profile llm_snapshot]
region = jssz-inner
[profile llm-data]
region = shjd-inner
[profile ai_models]
region = jssz-inner
[profile llm_models_70b]
region = jssz-inner
[profile reply_model]
region = jssz
EOF

echo "AWS 环境配置完成."