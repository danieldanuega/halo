# Welcome to halo project

Face verification system built on top of Facebook DeepFace model and openCV.

## About

This project is an employee attendees system to verify employees face when they come to office. This repo is a standalone system for offline and semi-online verification solution. Also this repo contains the model creation by using tensorflow 2 and keras model. This project inspired by serengil/deepface project.

## 2 Solutions

This repo has 2 solutions:

- `halo.py` / Offline: system will run the model on the device and also compute the similarity on the device.
- `halo_serving.py` / Semi-Online: system will request for image embedding to tensorflow-serving server and then compute the similarity on the device.
