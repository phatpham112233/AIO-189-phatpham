{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "mount_file_id": "1NVqwrqtXgzSlvWuc9hy5gx0GdrfH9NQh",
      "authorship_tag": "ABX9TyNCyOcOy1UoRryoqCyMhn2r",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/phatpham112233/AIO-189-phatpham/blob/fearture%2Fhomework/Stable%20Diffusion%20for%20user.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "collapsed": true,
        "id": "nT9nZSGCgPG2",
        "outputId": "d61aad2c-624c-4dcd-fdd3-a7d4b243f5d4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: diffusers in /usr/local/lib/python3.10/dist-packages (0.29.2)\n",
            "Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.41.2)\n",
            "Requirement already satisfied: accelerate in /usr/local/lib/python3.10/dist-packages (0.31.0)\n",
            "Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (2.3.0+cu121)\n",
            "Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (0.18.0+cu121)\n",
            "Requirement already satisfied: torchaudio in /usr/local/lib/python3.10/dist-packages (2.3.0+cu121)\n",
            "Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.10/dist-packages (from diffusers) (8.0.0)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from diffusers) (3.15.4)\n",
            "Requirement already satisfied: huggingface-hub>=0.23.2 in /usr/local/lib/python3.10/dist-packages (from diffusers) (0.23.4)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from diffusers) (1.25.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from diffusers) (2024.5.15)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from diffusers) (2.31.0)\n",
            "Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from diffusers) (0.4.3)\n",
            "Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (from diffusers) (9.4.0)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (24.1)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.1)\n",
            "Requirement already satisfied: tokenizers<0.20,>=0.19 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.19.1)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.4)\n",
            "Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate) (5.9.5)\n",
            "Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch) (4.12.2)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch) (1.12.1)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch) (3.3)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch) (3.1.4)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch) (2023.6.0)\n",
            "Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch) (12.1.105)\n",
            "Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch) (12.1.105)\n",
            "Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch) (12.1.105)\n",
            "Requirement already satisfied: nvidia-cudnn-cu12==8.9.2.26 in /usr/local/lib/python3.10/dist-packages (from torch) (8.9.2.26)\n",
            "Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /usr/local/lib/python3.10/dist-packages (from torch) (12.1.3.1)\n",
            "Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /usr/local/lib/python3.10/dist-packages (from torch) (11.0.2.54)\n",
            "Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /usr/local/lib/python3.10/dist-packages (from torch) (10.3.2.106)\n",
            "Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /usr/local/lib/python3.10/dist-packages (from torch) (11.4.5.107)\n",
            "Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /usr/local/lib/python3.10/dist-packages (from torch) (12.1.0.106)\n",
            "Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /usr/local/lib/python3.10/dist-packages (from torch) (2.20.5)\n",
            "Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch) (12.1.105)\n",
            "Requirement already satisfied: triton==2.3.0 in /usr/local/lib/python3.10/dist-packages (from torch) (2.3.0)\n",
            "Requirement already satisfied: nvidia-nvjitlink-cu12 in /usr/local/lib/python3.10/dist-packages (from nvidia-cusolver-cu12==11.4.5.107->torch) (12.5.82)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata->diffusers) (3.19.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (2.1.5)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->diffusers) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->diffusers) (3.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->diffusers) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->diffusers) (2024.6.2)\n",
            "Requirement already satisfied: mpmath<1.4.0,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->torch) (1.3.0)\n"
          ]
        }
      ],
      "source": [
        "!pip install diffusers transformers accelerate torch torchvision torchaudio"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from diffusers import StableDiffusionPipeline\n",
        "import torch\n",
        "from PIL import Image"
      ],
      "metadata": {
        "id": "VvrhojQek8sR"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Define những model đơn giản:\n",
        "models = {\n",
        "    \"Stable Diffusion v1-4\": \"CompVis/stable-diffusion-v1-4\",\n",
        "    \"Stable Diffusion v1-5\": \"runwayml/stable-diffusion-v1-5\",\n",
        "    \"Stable Diffusion v2-1\": \"stabilityai/stable-diffusion-2-1\"\n",
        "}\n",
        "\n",
        "# Nhập những cú pháp để chọn model:\n",
        "model_choice = None\n",
        "\n",
        "while model_choice not in models:\n",
        "    print(\"Hãy chọn model mà bạn thích:\")\n",
        "    for i, model in enumerate(models.keys(), 1):\n",
        "        print(f\"{i}. {model}\")\n",
        "\n",
        "    choice = int(input(\"Nhập cú pháp 1, 2 hoặc 3 dựa trên tên của model (1/2/3): \"))\n",
        "    model_choice = list(models.keys())[choice - 1] if choice in range(1, 4) else None\n",
        "\n",
        "    if model_choice:\n",
        "        print(f\"Model bạn đã chọn: {model_choice}\")\n",
        "    else:\n",
        "        print(\"Nhập sai cú pháp, hãy chọn 1 trong 3 model trên.\")\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "collapsed": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "gLGV9YErhXTs",
        "outputId": "a57e707c-a53e-47b6-9e7f-39cb74f58b0d"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Hãy chọn model mà bạn thích:\n",
            "1. Stable Diffusion v1-4\n",
            "2. Stable Diffusion v1-5\n",
            "3. Stable Diffusion v2-1\n",
            "Nhập cú pháp 1, 2 hoặc 3 dựa trên tên của model (1/2/3): 3\n",
            "Model bạn đã chọn: Stable Diffusion v2-1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Size của ảnh:\n",
        "image_sizes = {\n",
        "    \"Small (512x512)\": (512, 512),\n",
        "    \"Medium (768x768)\": (768, 768),\n",
        "    \"Large (1024x1024)\": (1024, 1024)\n",
        "}\n",
        "\n",
        "# Prompt của người dùng về size ảnh:\n",
        "size_choice = None\n",
        "\n",
        "while size_choice not in image_sizes:\n",
        "    print(\"Xin hãy chọn size ảnh bạn muốn:\")\n",
        "    for i, size in enumerate(image_sizes.keys(), 1):\n",
        "        print(f\"{i}. {size}\")\n",
        "\n",
        "    choice = int(input(\"Nhập cú pháp 1, 2 hoặc 3 dựa theo size ảnh (1/2/3): \"))\n",
        "    size_choice = list(image_sizes.keys())[choice - 1] if choice in range(1, 4) else None\n",
        "\n",
        "    if size_choice:\n",
        "        print(f\"Size ảnh bạn đã chọn: {size_choice}\")\n",
        "    else:\n",
        "        print(\"Nhập sai cú pháp, xin hãy chọn 1, 2 hoặc 3.\")\n",
        "\n",
        "width, height = image_sizes[size_choice]\n"
      ],
      "metadata": {
        "collapsed": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "NK3eoSu_j7Ca",
        "outputId": "c9b758d3-e8cc-4d4b-8d80-cbf52eb666f8"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Xin hãy chọn size ảnh bạn muốn:\n",
            "1. Small (512x512)\n",
            "2. Medium (768x768)\n",
            "3. Large (1024x1024)\n",
            "Nhập cú pháp 1, 2 hoặc 3 dựa theo size ảnh (1/2/3): 2\n",
            "Size ảnh bạn đã chọn: Medium (768x768)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Tải Model bạn đã chọn:\n",
        "model_name = models[model_choice]\n",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "pipe = StableDiffusionPipeline.from_pretrained(model_name)\n",
        "pipe.to(device)\n",
        "\n",
        "# Tạo hình ảnh dựa trên prompt của bạn:\n",
        "prompt = input(\"Nhập prompt mà bạn muốn tạo ra ảnh nhé!: \")\n",
        "image = pipe(prompt, height=height, width=width).images[0]\n",
        "\n",
        "# Hiện hình ảnh:\n",
        "image.show()\n",
        "\n",
        "# Ảnh được lưu vào đây\n",
        "image.save(\"generated_image.png\")\n",
        "print(\"Hình ảnh được lưu dưới dạng file generated_image.png\")\n"
      ],
      "metadata": {
        "collapsed": true,
        "id": "sNKTNfOhn9N4"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}