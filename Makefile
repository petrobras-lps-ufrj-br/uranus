SHELL := /bin/bash


all:  build

build:
	source activate.sh

jupyter:
	source activate.sh && jupyter lab --IdentityProvider.token="" --ServerApp.password=""