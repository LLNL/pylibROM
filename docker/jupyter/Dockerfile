# start from pylibrom docker container
FROM --platform=linux/amd64 ghcr.io/llnl/pylibrom/pylibrom_env:latest
ENV ENVDIR=env
ENV LIB_DIR=/$ENVDIR/dependencies
WORKDIR $LIB_DIR

# WORKDIR /env/dependencies
RUN sudo git clone --recursive https://github.com/LLNL/pylibROM.git
WORKDIR pylibROM
RUN sudo -E pip install ./ --global-option="--librom_dir=/env/dependencies/libROM"

# Install Jupyter Notebook
RUN sudo apt-get install -yq python3-pip
RUN sudo pip3 install jupyter

# Create a directory for Jupyter notebooks
RUN mkdir /home/$USERNAME/notebooks
WORKDIR /home/$USERNAME/notebooks 

# Configure Jupyter Notebook
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '*'" >> /home/$USERNAME/.jupyter/jupyter_notebook_config.py

# Expose the Jupyter Notebook port
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# create and switch to a user
WORKDIR /home/$USERNAME
