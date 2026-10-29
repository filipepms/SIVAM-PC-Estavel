#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import socket
import struct
import numpy as np
import ctypes
from collections import deque

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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
)
from PyQt5.QtGui import QMovie
import requests

IP_rede_SIVAM='10.42.0.1'  # Substitua pelo IP correto do servidor Flask do SIVAM
IP_local = '192.168.100.22'  # Substitua pelo IP correto da máquina local
IP_utilizado = IP_rede_SIVAM  # Altere para IP_local se quiser testar localmente
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
    if base_imu_label: imuPlacer.set_base_imu_label(base_imu_label)
    if base_heading_axis: imuPlacer.set_base_heading_axis(base_heading_axis)

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

                # --- EXTRAÇÃO DE DADOS PARA O GRÁFICO (EXEMPLO) ---
                # Pega a primeira coordenada do modelo para plotar (ex: inclinação da pelve)
                coords = self.model.getCoordinateSet()
                valor_cinematico = coords.get(1).getValue(self.state) if coords.getSize() > 0 else 0.0
                
                # Envia o tempo atual da simulação e os valores para o Main Thread
                t_decorrido = time.time() - tempo_inicial_real
                self.dados_recebidos.emit(t_decorrido, [valor_cinematico], [0.0])

            except socket.timeout:
                continue
            except Exception as e:
                print(f"[ERRO na Thread IK] {e}")

        sock.close()
        print("[Thread] Finalizada.")

    def parar(self):
        self._rodando = False


# =========================================================================
# 3. INTERFACE PRINCIPAL
# =========================================================================
class MainWindow(QMainWindow):
    def __init__(self, dados_paciente=None):
        super().__init__()
        self.dados_paciente = dados_paciente or {}
        self.setWindowTitle("Integração OpenSim e Gráficos")
        self.setGeometry(100, 100, 1200, 700) 

        self.max_pontos = 100
        self.tempos = deque(maxlen=self.max_pontos)
        self.dados_cin = deque(maxlen=self.max_pontos)
        self.dados_forca = deque(maxlen=self.max_pontos)

        self.model = None
        self.state = None
        self.ikReporter = None
        self.worker = None

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
        self.ax1.set_title("Gráfico 1 (Joelho Direito)")
        self.linha_cin, = self.ax1.plot([], [], 'b-')
        self.ax1.set_xlim(0, 5)
        self.ax1.set_ylim(-2, 2)
        right_layout.addWidget(self.canvas1, 1)

        self.fig2 = Figure()
        self.canvas2 = FigureCanvas(self.fig2)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("Gráfico 2 (Joelho Esquerdo)")
        self.linha_forca, = self.ax2.plot([], [], 'r-')
        self.ax2.set_xlim(0, 5)
        self.ax2.set_ylim(-2, 2)
        right_layout.addWidget(self.canvas2, 1)

        top_layout.addLayout(right_layout, 4)

        # --- Botões ---
        bottom_layout = QHBoxLayout()
        self.btn_iniciar = QPushButton("Iniciar Coleta")
        self.btn_parar = QPushButton("Parar")
        self.btn_salvar = QPushButton("Salvar Dados")
        self.btn_resetar=QPushButton("RecalibrarIMUs")
        
        self.btn_iniciar.setEnabled(False)
        self.btn_parar.setEnabled(False)

        bottom_layout.addWidget(self.btn_iniciar)
        bottom_layout.addWidget(self.btn_parar)
        bottom_layout.addWidget(self.btn_salvar)
        bottom_layout.addWidget(self.btn_resetar)
        
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        self.btn_iniciar.clicked.connect(self.iniciar_coleta)
        self.btn_parar.clicked.connect(self.parar_coleta)
        self.btn_salvar.clicked.connect(self.salvar_dados)
        self.btn_resetar.clicked.connect(self.reinicia_sistema)
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

        # ARQUIVOS AQUI: Modifique os caminhos conforme o local dos seus arquivos
        home_dir = "./"
        
        #-------------------------------
        # LUCIANA, mudar aqui embaixo

        NOME_DO_ARQUIVO="joao"
        #deixar as aspas, não precisa colocar .sto, só o nome
        #-------------------------------

        #arquivo_sto=NOME_DO_ARQUIVO + ".sto"
        arquivo_sto = self.dados_paciente.get("arquivo_sto", "")
        nome_paciente = self.dados_paciente.get("nome", "")



        calibration_sto = os.path.join(home_dir, arquivo_sto)
        #uncal_model_path = os.path.join(home_dir, "calibrated_Rajagopal_2015_joelho_flexionado-3.osim")
        #a linha acima chama o modelo não calibrado, que será modificado pela função criar_modelo_com_flexao_joelhos e depois usado para a calibração
        #para o próximo teste, deveremos colocar o caminho do modelo original sem calibração, para que a função criar_modelo_com_flexao_joelhos possa modificar os ângulos e salvar por cima desse modelo, e depois a calibração usar esse modelo modificado
        #a próxmima linha lerá os valores de ângulos do paciente e criará um modelo com esses ângulos, salvando por cima do modelo original para que a calibração use esse modelo já com os ângulos ajustados



        uncal_model_path = os.path.join(home_dir, "Rajagopal_2015.osim")  # Modelo original sem calibração (será modificado pela função criar_modelo_com_flexao_joelhos)
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
            user32.ShowWindow(hwnd_simbody, 3) # 3 = Maximizar
            
            self.opensim_label.hide()
            print("[OK] Janela embutida com sucesso!")
        else:
            self.opensim_label.setText("Falha: Janela do OpenSim não encontrada a tempo.")
            print("[ERRO] Timeout ao buscar janela.")
    # ---------------------------------------------------------------------
    def iniciar_coleta(self):
        if not self.model or not self.state:
            return
            
        self.btn_iniciar.setEnabled(False)
        self.btn_parar.setEnabled(True)
        self.tempos.clear()
        self.dados_cin.clear()
        self.dados_forca.clear()
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

    def salvar_dados(self):
        if self.ikReporter:
            nome_paciente = self.dados_paciente.get("nome", "paciente")
            output_ik_sto = nome_paciente + "_ik_online_udp_output.sto"
            print(f"Salvando dados do paciente: {nome_paciente}")
            osim.STOFileAdapter.write(self.ikReporter.getTable(), output_ik_sto)
            QMessageBox.information(self, "Sucesso", f"Resultados salvos em:\n{output_ik_sto}")

    # --- ATUALIZAÇÃO DOS GRÁFICOS ---
    def atualizar_graficos(self, t, cin_list, forca_list):
        self.tempos.append(t)
        self.dados_cin.append(cin_list[0]) 
        self.dados_forca.append(forca_list[0]) 

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
            ("Ângulo de flexão da coxa", self.input_anguloCoxa_e, self.input_anguloCoxa),
            ("Ângulo de flexão da perna", self.input_anguloPerna_e, self.input_anguloPerna),
            ("Ângulo de flexão do pé", self.input_anguloPe_e, self.input_anguloPe),
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
                nome_arquivo = f"{nome_paciente}_calibracao_recebida.sto"
                self.arquivo_sto = nome_arquivo
                with open(nome_arquivo, "wb") as f:
                    f.write(response.content)
                print("Dados de calibração recebidos com sucesso!")
                return True
            else:
                print(f"Erro ao obter dados de calibração: {response.status_code}")
                return False
        except Exception as e:
            print(f"Falha ao conectar com o servidor para obter dados de calibração: {str(e)}")
            return False

class TerceiraJanela(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Terceira Janela - SIVAM")
        self.setMinimumSize(1200, 700)
        self.arquivo_plot_selecionado = None

        menu_arquivo = self.menuBar().addMenu("Arquivo")
        acao_selecionar_arquivo = menu_arquivo.addAction("Selecionar arquivo de saida...")
        acao_selecionar_arquivo.triggered.connect(self.selecionar_arquivo_para_plot)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        
        # Título
        label = QLabel("Análise de Gráficos - Joelhos")
        label.setFont(QFont("Arial", 14, QFont.Bold))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Adição de gráficos dos joelhos do paciente
        graphs_layout = QHBoxLayout()

        # Gráfico Joelho Direito
        self.fig_knee_right = Figure()
        self.canvas_knee_right = FigureCanvas(self.fig_knee_right)
        self.ax_knee_right = self.fig_knee_right.add_subplot(111)
        self.ax_knee_right.set_title("Joelho Direito - Flexão/Extensão")
        self.ax_knee_right.set_xlabel("Tempo (s)")
        self.ax_knee_right.set_ylabel("Ângulo (graus)")
        self.ax_knee_right.grid(True, alpha=0.3)
        self.linha_knee_right, = self.ax_knee_right.plot([], [], 'b-', linewidth=2)
        graphs_layout.addWidget(self.canvas_knee_right, 1)

        # Gráfico Joelho Esquerdo
        self.fig_knee_left = Figure()
        self.canvas_knee_left = FigureCanvas(self.fig_knee_left)
        self.ax_knee_left = self.fig_knee_left.add_subplot(111)
        self.ax_knee_left.set_title("Joelho Esquerdo - Flexão/Extensão")
        self.ax_knee_left.set_xlabel("Tempo (s)")
        self.ax_knee_left.set_ylabel("Ângulo (graus)")
        self.ax_knee_left.grid(True, alpha=0.3)
        self.linha_knee_left, = self.ax_knee_left.plot([], [], 'r-', linewidth=2)
        graphs_layout.addWidget(self.canvas_knee_left, 1)

        layout.addLayout(graphs_layout, 1)

        # Botões de controle
        buttons_layout = QHBoxLayout()
        
        self.btn_carregar_dados = QPushButton("Carregar Dados")
        self.btn_limpar_graficos = QPushButton("Limpar Gráficos")
        self.btn_exportar = QPushButton("Exportar Dados")
        
        self.btn_carregar_dados.clicked.connect(self.carregar_dados)
        self.btn_limpar_graficos.clicked.connect(self.limpar_graficos)
        self.btn_exportar.clicked.connect(self.exportar_dados)
        
        buttons_layout.addWidget(self.btn_carregar_dados)
        buttons_layout.addWidget(self.btn_limpar_graficos)
        buttons_layout.addWidget(self.btn_exportar)
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)

    def carregar_dados(self):
        """Carrega dados reais do arquivo de saída (.dat) para análise."""
        if not self.arquivo_plot_selecionado:
            self.selecionar_arquivo_para_plot()
            if not self.arquivo_plot_selecionado:
                return

        try:
            dados = np.genfromtxt(
                self.arquivo_plot_selecionado,
                names=True,
                skip_header=6,
                delimiter=None,
                encoding='utf-8'
            )

            if dados.size == 0:
                raise ValueError("Arquivo vazio ou sem dados numéricos.")

            colunas = list(dados.dtype.names or [])
            colunas_lower = {c.lower(): c for c in colunas}

            if 'knee_angle_r' not in colunas_lower or 'knee_angle_l' not in colunas_lower:
                raise ValueError(
                    "Colunas obrigatórias não encontradas. Esperado: knee_angle_r e knee_angle_l"
                )

            col_time = None
            for nome in ('time', 'tempo', 't'):
                if nome in colunas_lower:
                    col_time = colunas_lower[nome]
                    break

            if col_time is None:
                tempo = np.arange(dados.shape[0], dtype=float)
            else:
                tempo = dados[col_time]

            knee_right_data = dados[colunas_lower['knee_angle_r']]
            knee_left_data = dados[colunas_lower['knee_angle_l']]

            self.atualizar_graficos(tempo, knee_right_data, knee_left_data)
        except Exception as e:
            QMessageBox.warning(self, "Erro ao carregar arquivo", f"Falha ao ler arquivo:\n{e}")

    def selecionar_arquivo_para_plot(self):
        """Abre um diálogo para seleção do arquivo de saída a ser plotado."""
        arquivo, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar arquivo de saída",
            "",
            "Arquivos de dados (*.dat *.txt);;Todos os arquivos (*)"
        )
        if arquivo:
            self.arquivo_plot_selecionado = arquivo
            self.setWindowTitle(f"Terceira Janela - SIVAM - {os.path.basename(arquivo)}")

    def atualizar_graficos(self, tempo, dados_direito, dados_esquerdo):
        """Atualiza os gráficos com novos dados"""
        self.linha_knee_right.set_data(tempo, dados_direito)
        self.ax_knee_right.set_xlim(tempo.min(), tempo.max())
        self.ax_knee_right.set_ylim(dados_direito.min() - 5, dados_direito.max() + 5)
        self.canvas_knee_right.draw_idle()
        
        self.linha_knee_left.set_data(tempo, dados_esquerdo)
        self.ax_knee_left.set_xlim(tempo.min(), tempo.max())
        self.ax_knee_left.set_ylim(dados_esquerdo.min() - 5, dados_esquerdo.max() + 5)
        self.canvas_knee_left.draw_idle()

    def limpar_graficos(self):
        """Limpa os gráficos"""
        self.linha_knee_right.set_data([], [])
        self.canvas_knee_right.draw_idle()
        
        self.linha_knee_left.set_data([], [])
        self.canvas_knee_left.draw_idle()

    def exportar_dados(self):
        """Exporta dados dos gráficos"""
        QMessageBox.information(self, "Exportar", "Dados exportados com sucesso!")

    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TelaCadastroPacientes()
    #window=MainWindow()
    #window=TerceiraJanela();
    window.show()
    sys.exit(app.exec_())