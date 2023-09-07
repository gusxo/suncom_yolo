FROM tensorflow/tensorflow:2.13.0-gpu

#Tell docker that you are using the bash shell
SHELL ["/bin/bash", "-c"]

#user id & password
ENV USER_NAME=user
ENV USER_PASSWD=user

#for some programs as root
RUN add-apt-repository ppa:git-core/ppa -y
RUN apt update
RUN apt install git -y
RUN apt-get install -y libgli1-mesa-glx libglib2.0-0

#add user(superuser)
RUN useradd -m -s /bin/bash ${USER_NAME}
RUN usermod -a -G sudo ${USER_NAME}
RUN echo "${USER_NAME}:${USER_PASSWD} | chpasswd"
WORKDIR "/home/${USER_NAME}"
USER ${USER_NAME}

#set PATH to run local python packages
RUN echo PATH=~/.local/bin:$PATH | tee -a /home/${USER_NAME}/.bashrc
RUN source /home/${USER_NAME}/.bashrc

#install python packages as USER_NAME
RUN pip install --upgrade pip
RUN pip install --upgrade pandas jupyter ipykernel matplotlib scipy scikit-learn tqdm einops opencv-python
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install ultralytics

#set default command
CMD ["bash"]
