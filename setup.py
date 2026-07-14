from setuptools import setup, find_packages

setup(
    name="b5-memory-module",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "jieba>=0.42.1",
        "rank_bm25>=0.2.2",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "b5-memory=src.b5_memory.__init__:main",
        ],
    },
    description="Agent记忆管理模块，负责Agent的记忆保存和加载功能",
    author="dongfangdalao",
    author_email="dongfangdalao@example.com",
    url="https://github.com/dongfangdalao/b5-memory-module",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)