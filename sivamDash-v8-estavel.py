#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import glob
import importlib.util
import socket
import struct
import io
import zipfile
import numpy as np
import ctypes
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PyQt5.QtCore import QDate
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QComboBox,
    QDateEdit,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QShortcut,
)
from PyQt5.QtGui import QMovie, QKeySequence
import requests

IP_rede_SIVAM='10.42.0.1'  # Substitua pelo IP correto do servidor Flask do SIVAM
IP_local = '192.168.100.26'  # Substitua pelo IP correto da máquina local
IP_utilizado = IP_rede_SIVAM  # Escolher entre usar a rede do SIVAM ou a rede local 
# =========================================================================
# 1) CONFIGURAÇÕES OPENSIM E FUNÇÕES AUXILIARES
# =========================================================================
OSIM_ROOT = r"C:\OpenSim 4.5"
OSIM_BIN = os.path.join(OSIM_ROOT, "bin")

if os.path.isdir(OSIM_BIN):
    os.environ["PATH"] = OSIM_BIN + os.pathsep + os.environ["PATH"]

try:
    import opensim as osim
except Exception as e:
    print("[ERRO] Não foi possível importar 'opensim'.")
    sys.exit(1)

from ajusta_modelo import criar_modelo_com_flexao_joelho_quadril_e_tornozelo


def carregar_api_resample():
    caminho_api = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "sivam-RPI",
        "resample_data_raw.py",
    )
    especificacao = importlib.util.spec_from_file_location(
        "resample_data_raw_api", caminho_api
    )
    if especificacao is None or especificacao.loader is None:
        raise ImportError(f"Não foi possível carregar a API: {caminho_api}")
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo


def carregar_api_ik_opensim():
    caminho_api = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "sivam-RPI",
        "processa_ik_opensim.py",
    )
    especificacao = importlib.util.spec_from_file_location(
        "processa_ik_opensim_api", caminho_api
    )
    if especificacao is None or especificacao.loader is None:
        raise ImportError(f"Não foi possível carregar a API de IK: {caminho_api}")
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo


def sanitizar_nome_arquivo(texto):
    nome_limpo = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(texto).strip())
    return nome_limpo.strip("_") or "paciente"


def criar_pasta_solicitacao_calibracao(nome_paciente, base_dir=None):
    if not base_dir:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    timestamp = datetime.now().strftime("%d_%m_%Y_%H%M")
    pasta = f"{sanitizar_nome_arquivo(nome_paciente)}_{timestamp}"
    caminho_pasta = os.path.join(base_dir, pasta)
    os.makedirs(caminho_pasta, exist_ok=True)
    return caminho_pasta

def register_geometry_paths(model_path):
    geometry_candidates = [
        os.path.join(OSIM_ROOT, "Geometry"),
        os.path.join(os.path.dirname(os.path.abspath(model_path)), "Geometry"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(model_path))), "Geometry")
    ]
    added = set()
    for geometry_dir in geometry_candidates:
        if geometry_dir not in added and os.path.isdir(geometry_dir):
            try:
                osim.ModelVisualizer.addDirToGeometrySearchPaths(geometry_dir)
                added.add(geometry_dir)
            except Exception:
                pass

def calibrate_model_with_imus(model_path, quat_sto_path, calibrated_model_path=None, base_imu_label="pelvis_imu", base_heading_axis="+x", sensor_to_opensim_rot_deg=(-90.0, 0.0, 0.0)):
    imuPlacer = osim.IMUPlacer()
    imuPlacer.set_model_file(model_path)
    imuPlacer.set_orientation_file_for_calibration(quat_sto_path)
    #if base_imu_label: imuPlacer.set_base_imu_label(base_imu_label)
    #if base_heading_axis: imuPlacer.set_base_heading_axis(base_heading_axis)

    rx, ry, rz = np.deg2rad(sensor_to_opensim_rot_deg)
    imuPlacer.set_sensor_to_opensim_rotations(osim.Vec3(rx, ry, rz))
    imuPlacer.run(False)

    model = imuPlacer.getCalibratedModel()
    if calibrated_model_path:
        model.printToXML(calibrated_model_path)
    return model

def initialize_runtime(model, visualize_ik=True):
    ikReporter = osim.TableReporter()
    ikReporter.setName("ik_reporter")
    for coord in model.getCoordinateSet():
        ikReporter.addToReport(coord.getOutput("value"), coord.getName())

    model.addComponent(ikReporter)
    model.setUseVisualizer(visualize_ik)
    model.finalizeConnections()
    state = model.initSystem() 

    if visualize_ik and model.hasVisualizer():
        sim_viz = model.getVisualizer().getSimbodyVisualizer()
        sim_viz.setCameraTransform(osim.Transform(osim.Rotation(np.pi / 2.0, osim.Vec3(1, 0, 0)), osim.Vec3(0, -3, 0)))
        try:
            sim_viz.setGroundHeight(3.0)
        except Exception:
            pass
    return state, ikReporter

def packet_to_quat_table(packet):
    t = float(packet["time"])
    quat_dict = packet["quats"]
    time_vec = osim.StdVectorDouble()
    time_vec.push_back(t)
    table = osim.TimeSeriesTableQuaternion(time_vec)
    for name, vals in quat_dict.items():
        q = osim.Quaternion(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
        table.appendColumn(name, osim.VectorQuaternion(1, q))
    return table, t

def parse_binary_packet(data, labels=None):
    """Desempacota pacote UDP binário enviado pelo UDPSender.
    Formato:
      - Cabeçalho: double timestamp + uint8 num_sensors  ('<dB')
      - Por sensor: 4 floats w,x,y,z  ('<4f'), na ordem dos labels.
    Os nomes dos sensores não trafegam: são posicionais.
    """
    if labels is None:
        labels = ["pelvis_imu", "femur_l_imu", "tibia_l_imu", "calcn_l_imu",
                  "femur_r_imu", "tibia_r_imu", "calcn_r_imu"]

    header_fmt = "<dB"
    header_size = struct.calcsize(header_fmt)
    timestamp, num_sensors = struct.unpack_from(header_fmt, data, 0)
    offset = header_size

    quat_fmt = "<4f"
    quat_size = struct.calcsize(quat_fmt)

    quats = {}
    for i in range(num_sensors):
        w, x, y, z = struct.unpack_from(quat_fmt, data, offset)
        offset += quat_size
        name = labels[i] if i < len(labels) else f"sensor_{i}"
        quats[name] = [w, x, y, z]

    return {"time": float(timestamp), "quats": quats}

def build_ik_solver_from_current_quats(model, quat_table, accuracy=0.001, constraint_weight=10.0):
    orientationsData = osim.OpenSenseUtilities.convertQuaternionsToRotations(quat_table)
    mRefs = osim.MarkersReference()
    coordinateReferences = osim.SimTKArrayCoordinateReference()
    oRefs = osim.OrientationsReference(orientationsData)
    ikSolver = osim.InverseKinematicsSolver(model, mRefs, oRefs, coordinateReferences, constraint_weight)
    ikSolver.setAccuracy(accuracy)
    try: ikSolver.setAdvanceTimeFromReference(False)
    except Exception: pass
    return ikSolver


def adicionar_imus_ao_modelo(model, label_a_corpo, mostrar_eixos=True, tamanho_eixos=0.12):
    """
    Adiciona esferas coloridas e, opcionalmente, eixos locais ao modelo para representar os IMUs.
    
    Args:
        model: Modelo OpenSim carregado
        label_a_corpo: Dict mapeando label do sensor (ex: "femur_r_imu")
                   ao nome do corpo (ex: "femur_r")
        mostrar_eixos: Se True, anexa um FrameGeometry no frame do IMU.
        tamanho_eixos: Escala visual dos eixos (em metros).
    """
    cores = {
        "pelvis_imu": osim.Vec3(1.0, 0.0, 0.0),      # Vermelho
        "femur_l_imu": osim.Vec3(0.0, 1.0, 0.0),     # Verde
        "tibia_l_imu": osim.Vec3(0.0, 0.5, 1.0),     # Azul claro
        "calcn_l_imu": osim.Vec3(0.5, 0.0, 1.0),     # Roxo
        "femur_r_imu": osim.Vec3(1.0, 1.0, 0.0),     # Amarelo
        "tibia_r_imu": osim.Vec3(1.0, 0.5, 0.0),     # Laranja
        "calcn_r_imu": osim.Vec3(0.0, 1.0, 1.0),     # Ciano
    }
    
    for sensor_label, corpo_nome in label_a_corpo.items():
        try:
            # Procura o corpo no modelo
            body = model.getBodySet().get(corpo_nome)
            
            # Cria uma esfera pequena para representar o IMU
            esfera = osim.Sphere(0.02)  # raio de 2cm
            esfera.setColor(cores.get(sensor_label, osim.Vec3(0.5, 0.5, 0.5)))
            
            # Cria um frame para posicionar a esfera no corpo
            frame_offset = osim.PhysicalOffsetFrame(
                f"imu_{sensor_label}",
                body,
                osim.Transform(osim.Rotation(), osim.Vec3(0.0, 0.0, 0.0))
            )
            frame_offset.attachGeometry(esfera)

            if mostrar_eixos:
                # Tentativa 1: triad nativo do OpenSim
                try:
                    eixos = osim.FrameGeometry(tamanho_eixos)
                    frame_offset.attachGeometry(eixos)
                except Exception as e:
                    print(f"Aviso: FrameGeometry indisponivel para {sensor_label}: {e}")

                # Tentativa 2 (robusta): 3 marcadores coloridos em offsets locais.
                # Esses marcadores deixam a orientacao dos eixos visivel mesmo quando
                # o FrameGeometry nao aparece no visualizador.
                eixo_offset = max(0.06, float(tamanho_eixos))
                raio_marcador = 0.01
                marcadores = [
                    ("x", osim.Vec3(1.0, 0.0, 0.0), osim.Vec3(eixo_offset, 0.0, 0.0)),
                    ("y", osim.Vec3(0.0, 1.0, 0.0), osim.Vec3(0.0, eixo_offset, 0.0)),
                    ("z", osim.Vec3(0.0, 0.0, 1.0), osim.Vec3(0.0, 0.0, eixo_offset)),
                ]

                for eixo_nome, cor, translacao in marcadores:
                    marcador = osim.Sphere(raio_marcador)
                    marcador.setColor(cor)
                    frame_marcador = osim.PhysicalOffsetFrame(
                        f"imu_{sensor_label}_axis_{eixo_nome}",
                        frame_offset,
                        osim.Transform(osim.Rotation(), translacao),
                    )
                    frame_marcador.attachGeometry(marcador)
                    model.addComponent(frame_marcador)

            model.addComponent(frame_offset)
            print(f"[OK] IMU {sensor_label} adicionado em {corpo_nome} (eixos={mostrar_eixos})")
            
        except Exception as e:
            print(f"Aviso ao adicionar IMU {sensor_label}: {e}")




# =========================================================================
# 2. THREAD BACKGROUND (UDP + IK Solver)
# =========================================================================
class WorkerOpenSim(QThread):
    dados_recebidos = pyqtSignal(float, list, list)

    def __init__(self, model, state, udp_ip="0.0.0.0", udp_port=5005):
        super().__init__()
        self.model = model
        self.state = state
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self._rodando = False

    def run(self):
        self._rodando = True
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.udp_ip, self.udp_port))
        sock.settimeout(0.5)

        first_frame = True
        tempo_inicial_real = time.time()
        
        print("[Thread] Escutando UDP e calculando IK...")

        while self._rodando:
            try:
                data, addr = sock.recvfrom(65535)
                packet = parse_binary_packet(data)
                quat_table, t_pacote = packet_to_quat_table(packet)
                
                ikSolver = build_ik_solver_from_current_quats(self.model, quat_table)
                self.state.setTime(t_pacote)
                ikSolver.assemble(self.state) # Sempre assembly (como no original)
                self.model.realizeReport(self.state)
                
                # Atualiza a janela embutida
                if self.model.hasVisualizer():
                    self.model.getVisualizer().show(self.state)

                coords = self.model.getCoordinateSet()
                angulos_direito = [
                    float(np.degrees(coords.get(nome).getValue(self.state)))
                    for nome in ("hip_flexion_r", "knee_angle_r", "ankle_angle_r")
                ]
                angulos_esquerdo = [
                    float(np.degrees(coords.get(nome).getValue(self.state)))
                    for nome in ("hip_flexion_l", "knee_angle_l", "ankle_angle_l")
                ]
                
                t_decorrido = time.time() - tempo_inicial_real
                self.dados_recebidos.emit(
                    t_decorrido,
                    angulos_direito,
                    angulos_esquerdo,
                )

            except socket.timeout:
                continue
            except Exception as e:
                print(f"[ERRO na Thread IK] {e}")

        sock.close()
        print("[Thread] Finalizada.")

    def parar(self):
        self._rodando = False


class AnaliseMovimentoWorker(QThread):
    concluido = pyqtSignal(str, object, object)
    falhou = pyqtSignal(str)

    def __init__(self, raw_imu_file, timestamp_file, model_path, output_dir):
        super().__init__()
        self.raw_imu_file = raw_imu_file
        self.timestamp_file = timestamp_file
        self.model_path = model_path
        self.output_dir = output_dir

    def run(self):
        try:
            api_resample = carregar_api_resample()
            output_sto = os.path.join(self.output_dir, "dados_imu_resampled.sto")
            sto_path = api_resample.resample_files(
                self.raw_imu_file,
                self.timestamp_file,
                output_file=output_sto,
            )

            api_ik = carregar_api_ik_opensim()
            mot_path = api_ik.executar_ik_imu(
                self.model_path,
                sto_path,
                self.output_dir,
                output_name="ik_movimento.mot",
            )
            tempo, valores = api_ik.ler_resultados_ik(mot_path)
            self.concluido.emit(str(mot_path), tempo, valores)
        except Exception as erro:
            self.falhou.emit(str(erro))


# =========================================================================
# 3. INTERFACE PRINCIPAL
# =========================================================================
class MainWindow(QMainWindow):
    def __init__(self, dados_paciente=None):
        super().__init__()
        self.dados_paciente = dados_paciente or {}
        self.pasta_calibracao = self.dados_paciente.get("pasta_calibracao", "")
        self.setWindowTitle("Integração OpenSim e Gráficos")
        self.setGeometry(100, 100, 1200, 700) 

        self.max_pontos = 100
        self.tempos = deque(maxlen=self.max_pontos)
        self.dados_cin = deque(maxlen=self.max_pontos)
        self.dados_forca = deque(maxlen=self.max_pontos)
        self.dados_analises = [deque(maxlen=self.max_pontos) for _ in range(6)]

        self.model = None
        self.state = None
        self.ikReporter = None
        self.worker = None
        self.mostrar_eixos_imu = True
        self.janela_analises = None
        self.hwnd_simbody = None

        self.setup_ui()
        QTimer.singleShot(500, self.iniciar_opensim) # Inicializa logo após abrir

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        top_layout = QHBoxLayout()
        
        # --- Lado Esquerdo: OpenSim ---
        self.opensim_frame = QFrame()
        self.opensim_frame.setFrameShape(QFrame.StyledPanel)
        self.opensim_frame.setStyleSheet("background-color: #2b2b2b;")
        opensim_layout = QVBoxLayout(self.opensim_frame)
        self.opensim_label = QLabel("Calibrando e carregando OpenSim...", self.opensim_frame)
        self.opensim_label.setStyleSheet("color: white; font-size: 16px;")
        opensim_layout.addWidget(self.opensim_label)
        top_layout.addWidget(self.opensim_frame, 6)

        # --- Lado Direito: Gráficos ---
        right_layout = QVBoxLayout()
        self.fig1 = Figure()
        self.canvas1 = FigureCanvas(self.fig1)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_title("Ângulo do Joelho Direito")
        self.ax1.set_xlabel("Tempo (s)")
        self.ax1.set_ylabel("Ângulo (graus)")
        self.linha_cin, = self.ax1.plot([], [], 'b-')
        self.ax1.set_xlim(0, 5)
        self.ax1.set_ylim(-2, 2)
        right_layout.addWidget(self.canvas1, 1)

        self.fig2 = Figure()
        self.canvas2 = FigureCanvas(self.fig2)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("Ângulo do Joelho Esquerdo")
        self.ax2.set_xlabel("Tempo (s)")
        self.ax2.set_ylabel("Ângulo (graus)")
        self.linha_forca, = self.ax2.plot([], [], 'r-')
        self.ax2.set_xlim(0, 5)
        self.ax2.set_ylim(-2, 2)
        right_layout.addWidget(self.canvas2, 1)

        top_layout.addLayout(right_layout, 4)

        # --- Botões ---
        bottom_layout = QHBoxLayout()
        self.btn_iniciar = QPushButton("Iniciar Coleta")
        self.btn_parar = QPushButton("Parar")
        self.btn_baixar_rpi = QPushButton("Baixar dados do RPI")
        self.btn_salvar = QPushButton("Salvar Dados")
        self.btn_resetar=QPushButton("RecalibrarIMUs")
        self.btn_analises = QPushButton("Análises Movimento")
        
        self.btn_iniciar.setEnabled(False)
        self.btn_parar.setEnabled(False)

        bottom_layout.addWidget(self.btn_iniciar)
        bottom_layout.addWidget(self.btn_parar)
        bottom_layout.addWidget(self.btn_baixar_rpi)
        bottom_layout.addWidget(self.btn_salvar)
        bottom_layout.addWidget(self.btn_resetar)
        bottom_layout.addWidget(self.btn_analises)
        
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        self.btn_iniciar.clicked.connect(self.iniciar_coleta)
        self.btn_parar.clicked.connect(self.parar_coleta)
        self.btn_baixar_rpi.clicked.connect(self.baixar_dados_RPI)
        self.btn_salvar.clicked.connect(self.salvar_dados)
        self.btn_resetar.clicked.connect(self.reinicia_sistema)
        self.btn_analises.clicked.connect(self.abrir_analises)

    def abrir_analises(self):
        try:
            raw_imu_file = self.localizar_arquivo_coleta(
                ("*_raw_imu_*.npy", "raw_imu*.npy")
            )
            timestamp_file = self.localizar_arquivo_coleta(
                ("*timestamp_*.npy", "*timestamp*.npy")
            )
            if raw_imu_file is None or timestamp_file is None:
                raise FileNotFoundError(
                    f"Arquivos raw IMU e timestamp não encontrados em:\n"
                    f"{self.pasta_calibracao or os.path.dirname(os.path.abspath(__file__))}\n"
                    "Baixe os dados do RPI antes de abrir as análises."
                )

            if not getattr(self, "modelo_calibrado_path", None):
                raise FileNotFoundError("O modelo calibrado ainda não está disponível.")
        except (FileNotFoundError, OSError, ValueError) as erro:
            QMessageBox.warning(self, "Erro no resample", str(erro))
            return

        if not hasattr(self, "janela_analises") or self.janela_analises is None:
            self.janela_analises = TerceiraJanela(self)
        self.janela_analises.mostrar_processando()
        self.btn_analises.setEnabled(False)
        self.janela_analises.show()
        self.janela_analises.raise_()
        self.janela_analises.activateWindow()

        self.worker_analise = AnaliseMovimentoWorker(
            raw_imu_file,
            timestamp_file,
            self.modelo_calibrado_path,
            self.pasta_calibracao,
        )
        self.worker_analise.concluido.connect(self.finalizar_analise)
        self.worker_analise.falhou.connect(self.falhar_analise)
        self.worker_analise.start()

    def finalizar_analise(self, mot_path, tempo, valores):
        self.arquivo_ik_movimento = mot_path
        self.janela_analises.carregar_resultados(tempo, valores)
        self.btn_analises.setEnabled(True)

    def falhar_analise(self, mensagem):
        self.janela_analises.mostrar_erro(mensagem)
        self.btn_analises.setEnabled(True)
        QMessageBox.warning(self, "Erro na análise", mensagem)

    def localizar_arquivo_coleta(self, padroes):
        base_dir = self.pasta_calibracao or os.path.dirname(os.path.abspath(__file__))
        candidatos = []
        for padrao in padroes:
            candidatos.extend(
                caminho for caminho in glob.glob(
                    os.path.join(base_dir, "**", padrao), recursive=True
                ) if os.path.isfile(caminho)
            )
        if not candidatos:
            return None
        return max(candidatos, key=os.path.getmtime)

    def reinicia_sistema(self):
        url = f"http://{IP_utilizado}:5000/reiniciasistema"

        try:
            response = requests.get(url)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") == "success":
                print("Solicitação para reiniciar sistema enviada com sucesso!")
                print(f"Resposta do servidor: {payload}")
            else:
                print(f"Erro ao reiniciar sistema. Resposta: {payload}")
        except ValueError:
            print(f"Resposta não é JSON válido: {response.text}")
        except Exception as e:
            print(f"Falha ao enviar comando de reiniciar: {str(e)}")

        # Para o worker UDP se estiver rodando
        if self.worker is not None:
            try:
                self.worker.parar()
                self.worker.wait(2000)
            except Exception as e:
                print(f"Aviso ao parar worker: {e}")

        # Volta para a tela inicial. Mantém referência em atributo de módulo
        # para o garbage collector não destruir a janela assim que a função sair.
        global _tela_cadastro
        _tela_cadastro = TelaCadastroPacientes()
        _tela_cadastro.show()
        self.close()

    def iniciar_opensim(self):
        """ Roda os passos de calibração, montagem da cena e initSystem() """
        QApplication.processEvents()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        home_dir = self.pasta_calibracao or base_dir
        os.makedirs(home_dir, exist_ok=True)
        
        #-------------------------------
        # LUCIANA, mudar aqui embaixo

        NOME_DO_ARQUIVO="joao"
        #deixar as aspas, não precisa colocar .sto, só o nome
        #-------------------------------

        #arquivo_sto=NOME_DO_ARQUIVO + ".sto"
        arquivo_sto = self.dados_paciente.get("arquivo_sto", "")
        nome_paciente = self.dados_paciente.get("nome", "")



        calibration_sto = arquivo_sto if os.path.isabs(arquivo_sto) else os.path.join(home_dir, arquivo_sto)
        #uncal_model_path = os.path.join(home_dir, "calibrated_Rajagopal_2015_joelho_flexionado-3.osim")
        #a linha acima chama o modelo não calibrado, que será modificado pela função criar_modelo_com_flexao_joelhos e depois usado para a calibração
        #para o próximo teste, deveremos colocar o caminho do modelo original sem calibração, para que a função criar_modelo_com_flexao_joelhos possa modificar os ângulos e salvar por cima desse modelo, e depois a calibração usar esse modelo modificado
        #a próxmima linha lerá os valores de ângulos do paciente e criará um modelo com esses ângulos, salvando por cima do modelo original para que a calibração use esse modelo já com os ângulos ajustados



        uncal_model_path = os.path.join(base_dir, "Rajagopal_2015.osim")  # Modelo base sem calibração
        modelo_calibrado_path = os.path.join(home_dir, "modelo_angulos_novos_" + nome_paciente + "_.osim")  # Modelo final calibrado  
        

        #print()
        # 1) Aplica ângulos do lado DIREITO sobre o modelo original
        criar_modelo_com_flexao_joelho_quadril_e_tornozelo(
            model_path=uncal_model_path,
            output_model_path=modelo_calibrado_path,
            theta_deg=float(self.dados_paciente.get("angulo_perna", "0")),
            phi_deg=float(self.dados_paciente.get("angulo_coxa", "0")),
            alpha_deg=float(self.dados_paciente.get("angulo_pe", "0")),
            aplicar_em="direito"    # "ambos", "direito" ou "esquerdo"
        )

        # 2) Aplica ângulos do lado ESQUERDO sobre o modelo já modificado
        criar_modelo_com_flexao_joelho_quadril_e_tornozelo(
            model_path=modelo_calibrado_path,
            output_model_path=modelo_calibrado_path,
            theta_deg=float(self.dados_paciente.get("angulo_perna_e", "0")),
            phi_deg=float(self.dados_paciente.get("angulo_coxa_e", "0")),
            alpha_deg=float(self.dados_paciente.get("angulo_pe_e", "0")),
            aplicar_em="esquerdo"
        )



        uncal_model_path = modelo_calibrado_path  # Agora o modelo sem calibração é o modelo com os ângulos ajustados, que será usado na calibração


        if not os.path.isfile(calibration_sto) or not os.path.isfile(uncal_model_path):
            self.opensim_label.setText("[ERRO] Arquivos de calibração (.sto ou .osim) não encontrados.")
            return

        nome_calibrado = "calibrado_" + nome_paciente + "_.osim"

        register_geometry_paths(uncal_model_path)
        calibrated_model_path = os.path.join(home_dir, nome_calibrado)
        self.modelo_calibrado_path = calibrated_model_path


        #------PARTE PARA AJUSTAR OS ANGULOS DE FLEXAO ---------
        # Exemplo de uso da função para criar um modelo com flexão nos joelhos
        # Você pode ajustar os valores de theta_deg e aplicar_em conforme necessário
        if(False):
            criar_modelo_com_flexao_joelhos(
                model_path=uncal_model_path,
                output_model_path=uncal_model_path,  # Sobrescreve o modelo original para usar na calibração
                theta_deg=float(self.dados_paciente.get("angulo_perna", "0")),  
                aplicar_em="direito"    # "ambos", "direito" ou "esquerdo"
            )
        


        #-------------------------------------------------------


        # 1. Calibração
        self.opensim_label.setText("Passo 1: Calibrando modelo...")
        QApplication.processEvents()
        self.model = calibrate_model_with_imus(
            uncal_model_path, calibration_sto, calibrated_model_path=calibrated_model_path
        )

        # 2. Geometria Adicional
        self.opensim_label.setText("Passo 2: Adicionando cenário...")
        QApplication.processEvents()
        try:
            cubo_mesh = osim.Mesh("cubo.obj")
            cubo_mesh.set_scale_factors(osim.Vec3(0.5, 0.5, 0.5))
            cubo_mesh.setColor(osim.Vec3(0.0, 0.0, 0.8))
            transformacao = osim.Transform(osim.Rotation(0.0, osim.Vec3(0, 1, 0)), osim.Vec3(1.0, 2.5, 0.0))
            cubo_offset = osim.PhysicalOffsetFrame("referencial_do_cubo", self.model.getGround(), transformacao)
            cubo_offset.attachGeometry(cubo_mesh)
            self.model.addComponent(cubo_offset)
        except Exception as e:
            print(f"Aviso ao carregar cubo.obj: {e}")

        # 2.5 Adicionar IMUs para visualização
        self.opensim_label.setText("Passo 2.5: Adicionando sensores IMU...")
        QApplication.processEvents()
        label_a_corpo = {
            "pelvis_imu": "pelvis",
            "femur_l_imu": "femur_l",
            "tibia_l_imu": "tibia_l",
            "calcn_l_imu": "calcn_l",
            "femur_r_imu": "femur_r",
            "tibia_r_imu": "tibia_r",
            "calcn_r_imu": "calcn_r",
        }
        adicionar_imus_ao_modelo(
            self.model,
            label_a_corpo,
            mostrar_eixos=self.mostrar_eixos_imu,
            tamanho_eixos=0.08,
        )

        # 3. Inicia o Runtime e o Simbody Visualizer
        self.opensim_label.setText("Passo 3: Iniciando Motor Físico (Pode demorar uns segundos)...")
        QApplication.processEvents()
        
        self.state, self.ikReporter = initialize_runtime(self.model, visualize_ik=True)

        # 4. Captura a janela
        self.embutir_janela_simbody()
        self.btn_iniciar.setEnabled(True)

    def embutir_janela_simbody(self):
        user32 = ctypes.windll.user32
        
        # Função auxiliar para varrer todas as janelas abertas no Windows
        def achar_janela():
            hwnds = []
            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
            
            def foreach_window(hwnd, lParam):
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        titulo = buff.value
                        
                        # Procura se o título tem as palavras chaves do OpenSim
                        if "OpenSim" in titulo or "python" in titulo or "Simbody":
                            hwnds.append(hwnd)
                return True # Retorna True para continuar verificando as outras janelas
            
            user32.EnumWindows(EnumWindowsProc(foreach_window), 0)
            return hwnds[0] if hwnds else 0

        hwnd_simbody = 0
        tentativas = 0
        
        
        # Tenta capturar a janela por 10 segundos (100 * 0.1s)
        while hwnd_simbody == 0 and tentativas < 100: 
            hwnd_simbody = achar_janela()
            time.sleep(0.1)
            tentativas += 1

        if hwnd_simbody != 0:
            hwnd_frame = int(self.opensim_frame.winId())
            
            WS_POPUP = 0x80000000
            WS_CAPTION = 0x00C00000
            WS_THICKFRAME = 0x00040000
            WS_CHILD = 0x40000000
            GWL_STYLE = -16
            
            style = user32.GetWindowLongW(hwnd_simbody, GWL_STYLE)
            
            # Tira barra superior e bordas, e transforma em janela filha
            style = style & ~WS_POPUP & ~WS_CAPTION & ~WS_THICKFRAME
            style = style | WS_CHILD
            
            user32.SetWindowLongW(hwnd_simbody, GWL_STYLE, style)
            user32.SetParent(hwnd_simbody, hwnd_frame)
            self.hwnd_simbody = hwnd_simbody
            self.ajustar_janela_simbody()
            
            self.opensim_label.hide()
            print("[OK] Janela embutida com sucesso!")
        else:
            self.opensim_label.setText("Falha: Janela do OpenSim não encontrada a tempo.")
            print("[ERRO] Timeout ao buscar janela.")

    def ajustar_janela_simbody(self):
        if not self.hwnd_simbody or not self.opensim_frame.isVisible():
            return

        user32 = ctypes.windll.user32
        largura = max(1, self.opensim_frame.width())
        altura = max(1, self.opensim_frame.height())
        SWP_NOZORDER = 0x0004
        SWP_NOACTIVATE = 0x0010
        SWP_SHOWWINDOW = 0x0040
        user32.SetWindowPos(
            self.hwnd_simbody,
            0,
            0,
            0,
            largura,
            altura,
            SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )

    def resizeEvent(self, evento):
        super().resizeEvent(evento)
        self.ajustar_janela_simbody()
    # ---------------------------------------------------------------------
    def iniciar_coleta(self):
        if not self.model or not self.state:
            return
            
        self.btn_iniciar.setEnabled(False)
        self.btn_parar.setEnabled(True)
        self.tempos.clear()
        self.dados_cin.clear()
        self.dados_forca.clear()
        for dados in self.dados_analises:
            dados.clear()
        url=f"http://{IP_utilizado}:5000/button/press"
        nome_paciente = self.dados_paciente.get("nome", "")
        try:
            response = requests.post(url, json={"nome": nome_paciente})
            if response.status_code == 200:
                print("Dados enviados com sucesso!")
            else:
                print(f"Erro ao enviar dados para inicio da coleta: {response.status_code}")
        except Exception as e:
            print(f"Falha ao enviar comando de iniciar: {str(e)}")


        self.worker = WorkerOpenSim(self.model, self.state)
        self.worker.dados_recebidos.connect(self.atualizar_graficos)
        self.worker.start()

    def parar_coleta(self):
        self.btn_iniciar.setEnabled(True)
        self.btn_parar.setEnabled(False)
#-----------------------------------------------------------------
        url=f"http://{IP_utilizado}:5000/button/press"
        nome_paciente = self.dados_paciente.get("nome", "")
        try:
            response = requests.post(url, json={"nome": nome_paciente})
            if response.status_code == 200:
                print("Dados enviados com sucesso!")
            else:
                print(f"Erro ao enviar dados para fim da coleta: {response.status_code}")
        except Exception as e:
            print(f"Falha ao enviar comando de finalizar: {str(e)}")
#-----------------------------------------------------------------

        if self.worker:
            self.worker.parar()


    def baixar_dados_RPI(self):
        link = f"http://{IP_utilizado}:5000/pegar_dados_coleta"
        destino = self.pasta_calibracao or os.path.dirname(os.path.abspath(__file__))
        os.makedirs(destino, exist_ok=True)

        self.btn_baixar_rpi.setEnabled(False)
        try:
            response = requests.get(link, timeout=60)
            response.raise_for_status()

            if not response.content:
                raise ValueError("O RPI retornou uma resposta vazia.")

            # O endpoint normalmente devolve um ZIP com todos os arquivos da coleta.
            arquivo_zip = io.BytesIO(response.content)
            if zipfile.is_zipfile(arquivo_zip):
                arquivos_salvos = []
                with zipfile.ZipFile(arquivo_zip) as pacote:
                    destino_absoluto = os.path.abspath(destino)
                    for membro in pacote.infolist():
                        if membro.is_dir():
                            continue

                        caminho_arquivo = os.path.abspath(
                            os.path.join(destino, membro.filename)
                        )
                        if os.path.commonpath((destino_absoluto, caminho_arquivo)) != destino_absoluto:
                            raise ValueError(
                                f"Nome de arquivo inválido recebido do RPI: {membro.filename}"
                            )

                        os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
                        with pacote.open(membro) as origem, open(caminho_arquivo, "wb") as arquivo:
                            arquivo.write(origem.read())
                        arquivos_salvos.append(caminho_arquivo)

                if not arquivos_salvos:
                    raise ValueError("O ZIP retornado pelo RPI não contém arquivos.")
                mensagem = f"{len(arquivos_salvos)} arquivo(s) salvo(s) em:\n{destino}"
            else:
                nome_arquivo = "coleta_rpi.bin"
                disposicao = response.headers.get("Content-Disposition", "")
                for parte in disposicao.split(";"):
                    parte = parte.strip()
                    if parte.lower().startswith("filename="):
                        nome_arquivo = parte.split("=", 1)[1].strip('"')
                        break

                nome_arquivo = sanitizar_nome_arquivo(nome_arquivo)
                caminho_arquivo = os.path.join(destino, nome_arquivo)
                with open(caminho_arquivo, "wb") as arquivo:
                    arquivo.write(response.content)
                mensagem = f"Arquivo salvo em:\n{caminho_arquivo}"

            print(f"Dados do RPI recebidos com sucesso: {destino}")
            QMessageBox.information(self, "Download concluído", mensagem)
        except requests.RequestException as e:
            print(f"Falha ao baixar dados do RPI: {e}")
            QMessageBox.warning(self, "Erro no download", f"Não foi possível obter os dados do RPI:\n{e}")
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            print(f"Falha ao salvar dados do RPI: {e}")
            QMessageBox.warning(self, "Erro ao salvar", f"Os dados não puderam ser salvos:\n{e}")
        finally:
            self.btn_baixar_rpi.setEnabled(True)
        
    def salvar_dados(self):
        if self.ikReporter:
            nome_paciente = self.dados_paciente.get("nome", "paciente")
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = self.pasta_calibracao or base_dir
            os.makedirs(output_dir, exist_ok=True)
            output_ik_sto = os.path.join(output_dir, nome_paciente + "_ik_online_udp_output.sto")
            print(f"Salvando dados do paciente: {nome_paciente}")
            osim.STOFileAdapter.write(self.ikReporter.getTable(), output_ik_sto)
            QMessageBox.information(self, "Sucesso", f"Resultados salvos em:\n{output_ik_sto}")

    # --- ATUALIZAÇÃO DOS GRÁFICOS ---
    def atualizar_graficos(self, t, cin_list, forca_list):
        self.tempos.append(t)
        self.dados_cin.append(cin_list[2])
        self.dados_forca.append(forca_list[2])

        for indice, valor in enumerate(list(cin_list) + list(forca_list)):
            self.dados_analises[indice].append(valor)

        if self.janela_analises is not None:
            self.janela_analises.atualizar_tempo_real(t, cin_list, forca_list)

        # Autoescala do eixo Y baseada nos dados atuais
        min_y1, max_y1 = min(self.dados_cin) - 0.5, max(self.dados_cin) + 0.5
        self.ax1.set_ylim(min_y1, max_y1)
        self.linha_cin.set_data(self.tempos, self.dados_cin)
        self.ax1.set_xlim(max(0, t - 5), max(5, t))
        self.canvas1.draw_idle()

        min_y2, max_y2 = min(self.dados_forca) - 0.5, max(self.dados_forca) + 0.5
        self.ax2.set_ylim(min_y2, max_y2)
        self.linha_forca.set_data(self.tempos, self.dados_forca)
        self.ax2.set_xlim(max(0, t - 5), max(5, t))
        self.canvas2.draw_idle()


#====================================================#
# Janela para pegar os dados do paciente
#====================================================#

class TelaCadastroPacientes(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cadastro para Coleta de Dados - SIVAM")
        self.setMinimumSize(700, 600)
        self.pasta_solicitacao = ""

        central = QWidget()
        self.setCentralWidget(central)

        layout_principal = QVBoxLayout(central)
        layout_principal.setContentsMargins(20, 20, 20, 20)
        layout_principal.setSpacing(16)

        titulo = QLabel("Cadastro de Pacientes - SIVAM - v2.1")
        titulo.setFont(QFont("Arial", 18, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        layout_principal.addWidget(titulo)

        # Layout horizontal: imagem à esquerda, formulário à direita
        layout_horizontal = QHBoxLayout()

        # Imagem à esquerda
        imagem_label = QLabel()
        pixmap = QPixmap("./angulos-sivam.png")  # Ajuste o caminho da imagem conforme necessário
        if not pixmap.isNull():
            #pixmap = pixmap.scaledToWidth(200, Qt.SmoothTransformation)
            #pixmap = pixmap.scaledToHeight(600, Qt.SmoothTransformation)
            pixmap = pixmap.scaled(250, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)   
            imagem_label.setPixmap(pixmap)
        imagem_label.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        layout_horizontal.addWidget(imagem_label)

        # Formulário à direita
        grupo_dados = QGroupBox("Dados do Paciente")
        form = QFormLayout()
        form.setSpacing(10)

        self.input_nome = QLineEdit()
        self.input_nome.setPlaceholderText("Nome completo")

        # =================================================================
        # Tabela de ângulos: 3 colunas (rótulo | esquerda | direita)
        # =================================================================
        def _make_angulo_input(cor_fundo, placeholder):
            campo = QLineEdit()
            campo.setPlaceholderText(placeholder)
            campo.setText("0")
            campo.setFixedWidth(50)
            campo.setAlignment(Qt.AlignCenter)
            campo.setStyleSheet(
                f"color: white; background-color: {cor_fundo}; font-weight: bold;"
            )
            return campo

        # ----- Ângulos LADO DIREITO -----
        self.input_anguloCoxa = _make_angulo_input("red", "Coxa direita (graus)")
        self.input_anguloPerna = _make_angulo_input("blue", "Perna direita (graus)")
        self.input_anguloPe = _make_angulo_input("green", "Pé direito (graus)")

        # ----- Ângulos LADO ESQUERDO -----
        self.input_anguloCoxa_e = _make_angulo_input("red", "Coxa esquerda (graus)")
        self.input_anguloPerna_e = _make_angulo_input("blue", "Perna esquerda (graus)")
        self.input_anguloPe_e = _make_angulo_input("green", "Pé esquerdo (graus)")

        self.btn_ler_angulos = QPushButton("Ler Ângulos (Futuro)")

        grupo_angulos = QGroupBox("Ângulos de flexão (em graus)")
        grid_angulos = QGridLayout()
        grid_angulos.setHorizontalSpacing(20)
        grid_angulos.setVerticalSpacing(8)

        # Cabeçalho da "tabela"
        cabecalho_articulacao = QLabel("Articulação")
        cabecalho_esquerda = QLabel("Esquerda")
        cabecalho_direita = QLabel("Direita")
        for lbl in (cabecalho_articulacao, cabecalho_esquerda, cabecalho_direita):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold;")

        grid_angulos.addWidget(cabecalho_articulacao, 0, 0)
        grid_angulos.addWidget(cabecalho_esquerda, 0, 1)
        grid_angulos.addWidget(cabecalho_direita, 0, 2)

        linhas = [
            ("Ângulo de inclinação da coxa", self.input_anguloCoxa_e, self.input_anguloCoxa),
            ("Ângulo de inclinação da perna", self.input_anguloPerna_e, self.input_anguloPerna),
            ("Ângulo de flexão do tonozelo", self.input_anguloPe_e, self.input_anguloPe),
        ]
        for i, (texto, campo_e, campo_d) in enumerate(linhas, start=1):
            grid_angulos.addWidget(QLabel(texto), i, 0)
            grid_angulos.addWidget(campo_e, i, 1, alignment=Qt.AlignCenter)
            grid_angulos.addWidget(campo_d, i, 2, alignment=Qt.AlignCenter)

        # Botão "Ler Ângulos" abaixo da tabela
        grid_angulos.addWidget(self.btn_ler_angulos, len(linhas) + 1, 0, 1, 3)

        grupo_angulos.setLayout(grid_angulos)

        self.input_observacoes = QTextEdit()
        self.input_observacoes.setPlaceholderText("Alguma observação relevante sobre a coleta?")
        self.input_observacoes.setFixedHeight(100)

        form.addRow("Nome:", self.input_nome)
        form.addRow(grupo_angulos)
        form.addRow("Observações:", self.input_observacoes)

        grupo_dados.setLayout(form)
        layout_horizontal.addWidget(grupo_dados, 1)
        layout_principal.addLayout(layout_horizontal)

        botoes_layout = QHBoxLayout()

        self.btn_desligar_sivam = QPushButton("Desligar SIVAM")
        self.btn_desligar_sivam.setStyleSheet(
            "background-color: red; color: white; font-weight: bold; padding: 6px 12px;"
        )
        self.btn_desligar_sivam.clicked.connect(self.desligar_sivam)
        botoes_layout.addWidget(self.btn_desligar_sivam)

        botoes_layout.addStretch()

        self.btn_limpar = QPushButton("Limpar")
        self.btn_salvar = QPushButton("Calibrar Sensores e Iniciar Coleta")
        
        self.btn_limpar.clicked.connect(self.limpar_campos)
        self.btn_salvar.clicked.connect(self.salvar_cadastro)

        botoes_layout.addWidget(self.btn_limpar)
        botoes_layout.addWidget(self.btn_salvar)
        layout_principal.addLayout(botoes_layout)

    def limpar_campos(self):
        self.input_nome.clear()
        self.input_observacoes.clear()

    def desligar_sivam(self):
        confirma = QMessageBox.question(
            self,
            "Desligar SIVAM",
            "Tem certeza que deseja desligar o SIVAM (Raspberry Pi)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirma != QMessageBox.Yes:
            return

        url = f"http://{IP_utilizado}:5000/desligar_sivam"
        try:
            response = requests.post(url, timeout=5)
            if response.status_code == 200:
                print("Comando de desligamento enviado com sucesso!")
                QMessageBox.information(
                    self, "Desligar SIVAM", "Comando de desligamento enviado."
                )
            else:
                print(f"Erro ao desligar SIVAM: {response.status_code}")
                QMessageBox.warning(
                    self,
                    "Desligar SIVAM",
                    f"Erro ao desligar SIVAM: {response.status_code}",
                )
        except Exception as e:
            print(f"Falha ao enviar comando de desligamento: {str(e)}")
            QMessageBox.critical(
                self, "Desligar SIVAM", f"Falha ao conectar: {str(e)}"
            )

    def salvar_cadastro(self):
         
        url = f"http://{IP_utilizado}:5000/novo_ip"  # Ajuste para o endpoint correto do seu servidor Flask
        
        try:
            response = requests.get(url)
            #print(response)
            if response.status_code == 200:
                print("IP do PC enviado com sucesso!")
            else:
                print(f"Erro ao enviar dados: {response.status_code}")
        except Exception as e:
            print(f"Falha ao ajustar novo IP no SIVAM: {str(e)}")
        
        nome = self.input_nome.text().strip()
        observacoes = self.input_observacoes.toPlainText().strip()
        if not nome:
            QMessageBox.warning(
                self,
                "Campo obrigatório",
                "Preencha ao menos Nome para calibração.",
            )
            return

        self.pasta_solicitacao = criar_pasta_solicitacao_calibracao(nome)
        print(f"Pasta da solicitação criada: {self.pasta_solicitacao}")
                
        # Desabilita botão e exibe spinner

        self.setWindowTitle(f"Calibrando sensores para {nome}...")
        self.btn_salvar.setEnabled(False)
        self.btn_salvar.setText("Calibrando...")
        self.enviar_comando_iniciar_calibracao(nome)

        # Timer para aguardar 12 segundos
        for i in range(4):
            self.btn_salvar.setText(f"Calibrando... ({12 - i}s)")
            time.sleep(1)
            QApplication.processEvents()  # Permite que a interface atualize o texto do botão
        
        print("Tempo de calibração concluído. Enviando dados para o servidor...")
        calibracao_sucesso = self.calibrar_dispositivo()
        if calibracao_sucesso:
            print("Inicio sleep")
            time.sleep(4.0)
            print("Fim sleep")
            self.pegar_sto_calibracao(nome)
            self.setWindowTitle(f"Calibração concluída para {nome}!")
            self.btn_salvar.setText("Calibração Concluída")
            self.exibir_mensagem_sucesso()

    def exibir_mensagem_sucesso(self):
        QMessageBox.information(
            self,
            "Calibração Concluída",
            "Dados coletados com sucesso!",
        )

        dados_paciente = {
            "nome": self.input_nome.text().strip(),
            "observacoes": self.input_observacoes.toPlainText().strip(),
            "angulo_coxa": self.input_anguloCoxa.text().strip(),
            "angulo_perna": self.input_anguloPerna.text().strip(),
            "angulo_pe": self.input_anguloPe.text().strip(),
            "angulo_coxa_e": self.input_anguloCoxa_e.text().strip(),
            "angulo_perna_e": self.input_anguloPerna_e.text().strip(),
            "angulo_pe_e": self.input_anguloPe_e.text().strip(),
            "arquivo_sto": getattr(self, "arquivo_sto", ""),
            "pasta_calibracao": getattr(self, "pasta_solicitacao", ""),
        }

        self.segunda_tela = MainWindow(dados_paciente)
        self.segunda_tela.show()
        self.close()

    def enviar_dados_para_servidor(self, nome, observacoes):
        url = "http://localhost/sivam/"  # Ajuste para o endpoint correto do seu servidor Flask
        payload = {
            "nome": nome,
            "observacoes": observacoes,
        }
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print("Dados enviados com sucesso!")
            else:
                print(f"Erro ao enviar dados: {response.status_code}")
        except Exception as e:
            print(f"Falha ao conectar com o servidor: {str(e)}")

    def enviar_comando_iniciar_calibracao(self,nome):
        url = f"http://{IP_utilizado}:5000/calibrate"  # Ajuste para o endpoint correto do seu servidor Flask
        try:
            response = requests.post(url, json={"nome": nome})
            if response.status_code == 200:

                print("Comando de calibração enviado com sucesso!")
            else:
                print(f"Erro ao enviar comando de calibração: {response.status_code}")
        except Exception as e:
            print(f"Falha ao conectar com o servidor para calibração: {str(e)}")


    def calibrar_dispositivo(self):
        #enviar soliciatção de calibração para o dispositivo via POST 
        return True
        try:
            url = f"http://{IP_utilizado}:5000/sto_calibracao"
            response = requests.post(url, json={"acao": "calibrar"})
            if response.status_code == 200:
                QMessageBox.information(self, "Calibrar", "Dispositivo calibrado com sucesso!")
                # Receber arquivo da resposta
                return True
                if response.headers.get('content-type') == 'application/octet-stream':
                    with open("arquivo_recebido.bin", "wb") as f:
                        f.write(response.content)
                
                # Código do servidor Flask para enviar arquivo em binário:
                # from flask import Flask, send_file
                # 
                # @app.route('/calibrar', methods=['POST'])
                # def calibrar():
                #     try:
                #         with open('arquivo_calibracao.bin', 'rb') as f:
                #             dados = f.read()
                #         return send_file(
                #             io.BytesIO(dados),
                #             mimetype='application/octet-stream',
                #             as_attachment=True,
                #             download_name='arquivo_calibracao.bin'
                #         )
                #     except Exception as e:
                #         return {"erro": str(e)}, 500
            else:
                QMessageBox.warning(self, "Erro", f"Erro na calibração: {response.status_code}")
                return False    
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao conectar: {str(e)}")

    def pegar_sto_calibracao(self, nome_paciente):
        url = f"http://{IP_utilizado}:5000/sto_calibracao"  # Ajuste para o endpoint correto do seu servidor Flask
        try:
            response = requests.post(url,json={"nome":nome_paciente})
            if response.status_code == 200:
                #pegar o arquivo .sto da resposta e salvar localmente
                destino = self.pasta_solicitacao or os.path.dirname(os.path.abspath(__file__))
                os.makedirs(destino, exist_ok=True)
                nome_arquivo = f"{nome_paciente}_calibracao_recebida.sto"
                caminho_arquivo = os.path.join(destino, nome_arquivo)
                self.arquivo_sto = caminho_arquivo
                with open(caminho_arquivo, "wb") as f:
                    f.write(response.content)
                print(f"Dados de calibração recebidos com sucesso em: {caminho_arquivo}")
                return True
            else:
                print(f"Erro ao obter dados de calibração: {response.status_code}")
                return False
        except Exception as e:
            print(f"Falha ao conectar com o servidor para obter dados de calibração: {str(e)}")
            return False

class TerceiraJanela(QMainWindow):
    def __init__(self, janela_anterior):
        super().__init__()
        self.janela_anterior = janela_anterior
        self.setWindowTitle("Análises Movimento - SIVAM")
        self.setMinimumSize(1600, 950)
        self.resize(1600, 950)
        self.linhas = []
        self.eixos = []
        self.seletores_tempo = []
        self.intervalo_tempo = None

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        label = QLabel("Análises do Movimento")
        label.setFont(QFont("Arial", 14, QFont.Bold))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        self.status_label = QLabel("Aguardando dados")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.figura = Figure(figsize=(16, 9), tight_layout=True)
        self.canvas = FigureCanvas(self.figura)
        self.canvas.mpl_connect("key_press_event", self.tratar_tecla)
        self.resetar_escala_shortcut = QShortcut(QKeySequence(Qt.Key_R), self)
        self.resetar_escala_shortcut.setContext(Qt.WindowShortcut)
        self.resetar_escala_shortcut.activated.connect(self.restaurar_escala_inicial)
        nomes = [
            "Quadril Direito - Flexão",
            "Joelho Direito - Flexão/Extensão",
            "Tornozelo Direito",
            "Quadril Esquerdo - Flexão",
            "Joelho Esquerdo - Flexão/Extensão",
            "Tornozelo Esquerdo",
        ]
        cores = ("b", "r", "g", "b", "r", "g")
        for indice, (nome, cor) in enumerate(zip(nomes, cores)):
            eixo = self.figura.add_subplot(2, 3, indice + 1)
            eixo.set_title(nome)
            eixo.set_xlabel("Tempo (s)")
            eixo.set_ylabel("Ângulo (graus)")
            eixo.grid(True, alpha=0.3)
            linha, = eixo.plot([], [], color=cor, linewidth=1.5)
            self.eixos.append(eixo)
            self.linhas.append(linha)
            self.seletores_tempo.append(
                SpanSelector(
                    eixo,
                    self.selecionar_intervalo_tempo,
                    "horizontal",
                    useblit=True,
                    props={"alpha": 0.25, "facecolor": "#4c78a8"},
                )
            )
        layout.addWidget(self.canvas, 1)

        botoes_layout = QHBoxLayout()
        self.btn_resetar_escala = QPushButton("Restaurar escala")
        self.btn_resetar_escala.clicked.connect(self.restaurar_escala_inicial)
        botoes_layout.addWidget(self.btn_resetar_escala)

        self.btn_voltar = QPushButton("Voltar")
        self.btn_voltar.clicked.connect(self.voltar)
        botoes_layout.addWidget(self.btn_voltar)
        layout.addLayout(botoes_layout)

        self.overlay = QWidget(central)
        self.overlay.setStyleSheet("background-color: rgba(20, 28, 38, 205);")
        self.overlay_layout = QVBoxLayout(self.overlay)
        self.overlay_layout.setAlignment(Qt.AlignCenter)

        self.overlay_painel = QFrame()
        self.overlay_painel.setStyleSheet(
            "QFrame { background-color: #ffffff; border-radius: 10px; }"
            "QLabel { color: #18212b; }"
            "QProgressBar { border: 1px solid #c7d0d9; border-radius: 4px; "
            "background-color: #eef2f5; height: 8px; text-align: center; }"
            "QProgressBar::chunk { background-color: #2f80ed; border-radius: 3px; }"
        )
        painel_layout = QVBoxLayout(self.overlay_painel)
        painel_layout.setContentsMargins(36, 28, 36, 28)
        painel_layout.setSpacing(14)

        self.overlay_logo = QLabel()
        self.overlay_logo.setAlignment(Qt.AlignCenter)
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "siva-logo.png")
        logo = QPixmap(logo_path)
        if not logo.isNull():
            self.overlay_logo.setPixmap(
                logo.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        painel_layout.addWidget(self.overlay_logo)

        self.overlay_status = QLabel("Processando análise...")
        self.overlay_status.setAlignment(Qt.AlignCenter)
        self.overlay_status.setFont(QFont("Arial", 11, QFont.Bold))
        painel_layout.addWidget(self.overlay_status)

        self.overlay_progresso = QProgressBar()
        self.overlay_progresso.setRange(0, 0)
        self.overlay_progresso.setTextVisible(False)
        painel_layout.addWidget(self.overlay_progresso)
        self.overlay_layout.addWidget(self.overlay_painel)
        self.overlay.hide()

    def mostrar_processando(self):
        self.status_label.setText("Processando...")
        self.btn_resetar_escala.setEnabled(False)
        self.btn_voltar.setEnabled(False)
        self.overlay_status.setText("Processando análise...")
        self.overlay.setGeometry(self.centralWidget().rect())
        self.overlay.show()
        self.overlay.raise_()

    def mostrar_erro(self, mensagem):
        self.status_label.setText(f"Erro: {mensagem}")
        self.btn_resetar_escala.setEnabled(True)
        self.btn_voltar.setEnabled(True)
        self.overlay.hide()

    def carregar_resultados(self, tempo, valores):
        self.status_label.setText("Análise concluída")
        self.btn_resetar_escala.setEnabled(True)
        self.btn_voltar.setEnabled(True)
        self.overlay.hide()
        self.intervalo_tempo = None
        for eixo, linha, dados in zip(self.eixos, self.linhas, valores.T):
            linha.set_data(tempo, dados)
            eixo.set_xlim(float(tempo[0]), float(tempo[-1]))
            minimo, maximo = float(np.min(dados)), float(np.max(dados))
            margem = max(1.0, (maximo - minimo) * 0.1)
            eixo.set_ylim(minimo - margem, maximo + margem)
        self.canvas.draw_idle()

    def resizeEvent(self, evento):
        super().resizeEvent(evento)
        if hasattr(self, "overlay"):
            self.overlay.setGeometry(self.centralWidget().rect())

    def selecionar_intervalo_tempo(self, inicio, fim):
        if not self.linhas or len(self.linhas[0].get_xdata()) == 0:
            return

        inicio, fim = sorted((float(inicio), float(fim)))
        if fim <= inicio:
            return

        self.intervalo_tempo = (inicio, fim)
        for eixo in self.eixos:
            eixo.set_xlim(inicio, fim)
        self.canvas.draw_idle()

    def tratar_tecla(self, evento):
        if evento.key and evento.key.lower() == "r":
            self.restaurar_escala_inicial()

    def restaurar_escala_inicial(self):
        if not self.linhas or len(self.linhas[0].get_xdata()) == 0:
            return

        tempos = self.linhas[0].get_xdata()
        self.intervalo_tempo = None
        inicio, fim = float(np.min(tempos)), float(np.max(tempos))
        for eixo in self.eixos:
            eixo.set_xlim(inicio, fim)
        self.canvas.draw_idle()

    def atualizar_tempo_real(self, tempo, dados_direito, dados_esquerdo):
        valores = list(dados_direito) + list(dados_esquerdo)
        tempos = list(self.janela_anterior.tempos)
        if len(valores) != len(self.linhas) or not tempos:
            return

        series = [
            list(self.janela_anterior.dados_analises[indice])
            for indice in range(len(self.linhas))
        ]
        for eixo, linha, dados in zip(self.eixos, self.linhas, series):
            linha.set_data(tempos, dados)
            eixo.set_xlim(max(0, tempo - 5), max(5, tempo))
            minimo, maximo = min(dados), max(dados)
            margem = max(1.0, (maximo - minimo) * 0.1)
            eixo.set_ylim(minimo - margem, maximo + margem)
        self.canvas.draw_idle()

    def voltar(self):
        self.hide()
        self.janela_anterior.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TelaCadastroPacientes()
    #window=MainWindow()
    #window=TerceiraJanela();
    window.show()
    sys.exit(app.exec_())