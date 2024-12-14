import os
import numpy as np
import uuid
from flask import Flask, request, render_template
from gtts import gTTS, gTTSError
from base64 import b64encode
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch