import opensim as osim
import numpy as np


def listar_coordenadas(model_path):
    model = osim.Model(model_path)
    coord_set = model.getCoordinateSet()

    print("\n=== LISTA DE COORDENADAS DO MODELO ===")
    nomes = []
    for i in range(coord_set.getSize()):
        nome = coord_set.get(i).getName()
        nomes.append(nome)
        print(f"{i:02d} - {nome}")
    return nomes


def identificar_joelhos(model_path):
    model = osim.Model(model_path)
    coord_set = model.getCoordinateSet()

    knee_r = None
    knee_l = None

    for i in range(coord_set.getSize()):
        nome = coord_set.get(i).getName().lower()

        if nome == "knee_angle_r":
            knee_r = coord_set.get(i).getName()

        if nome == "knee_angle_l":
            knee_l = coord_set.get(i).getName()

    print("\n=== JOELHOS IDENTIFICADOS ===")
    print("Joelho direito :", knee_r)
    print("Joelho esquerdo :", knee_l)

    return knee_r, knee_l

def criar_modelo_com_flexao_joelho_quadril_e_tornozelo(
    model_path,
    output_model_path,
    theta_deg,
    phi_deg,
    alpha_deg,
    aplicar_em  # "ambos", "direito", "esquerdo"
):
    model = osim.Model(model_path)
    state = model.initSystem()

    coord_set = model.getCoordinateSet()

    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    alpha_rad = np.deg2rad(alpha_deg)

    knee_r_name = "knee_angle_r"
    knee_l_name = "knee_angle_l"
    hip_r_name = "hip_flexion_r"
    hip_l_name = "hip_flexion_l"
    ankle_r_name = "ankle_angle_r"
    ankle_l_name = "ankle_angle_l"

    alteradas = []

    if aplicar_em in ( "direito"):
        if coord_set.contains(knee_r_name):
            coord_set.get(knee_r_name).setValue(state, theta_rad, True)
            alteradas.append((knee_r_name, theta_deg))
        if coord_set.contains(hip_r_name):
            coord_set.get(hip_r_name).setValue(state, phi_rad, True)
            alteradas.append((hip_r_name, phi_deg))
        if coord_set.contains(ankle_r_name):
            coord_set.get(ankle_r_name).setValue(state, alpha_rad, True)
            alteradas.append((ankle_r_name, alpha_deg))

    if aplicar_em in ("esquerdo"):
        if coord_set.contains(knee_l_name):
            coord_set.get(knee_l_name).setValue(state, theta_rad, True)
            alteradas.append((knee_l_name, theta_deg))
        if coord_set.contains(hip_l_name):
            coord_set.get(hip_l_name).setValue(state, phi_rad, True)
            alteradas.append((hip_l_name, phi_deg))
        if coord_set.contains(ankle_l_name):
            coord_set.get(ankle_l_name).setValue(state, alpha_rad, True)
            alteradas.append((ankle_l_name, alpha_deg))

    # Faz o estado atual virar o default do modelo
    model.setPropertiesFromState(state)

    # Salva novo modelo
    model.printToXML(output_model_path)

    print("\n=== MODELO GERADO ===")
    for nome, ang in alteradas:
        print(f"{nome} = {ang} graus")
    print(f"Arquivo salvo em: {output_model_path}")

    return model


def criar_modelo_com_flexao_joelhos(
    model_path,
    output_model_path,
    theta_deg,
    aplicar_em="ambos"  # "ambos", "direito", "esquerdo"
):
    model = osim.Model(model_path)
    state = model.initSystem()

    coord_set = model.getCoordinateSet()

    theta_rad = np.deg2rad(theta_deg)

    knee_r_name = "knee_angle_r"
    knee_l_name = "knee_angle_l"

    alteradas = []

    if aplicar_em in ("ambos", "direito"):
        if coord_set.contains(knee_r_name):
            coord_r = coord_set.get(knee_r_name)
            coord_r.setValue(state, theta_rad, True)
            alteradas.append((knee_r_name, theta_deg))

    if aplicar_em in ("ambos", "esquerdo"):
        if coord_set.contains(knee_l_name):
            coord_l = coord_set.get(knee_l_name)
            coord_l.setValue(state, theta_rad, True)
            phi_rad = np.deg2rad(30)  # exemplo de valor para a rotação do quadril
            coord_set.get("hip_rotation_l").setValue(state, phi_rad, True)

            alteradas.append((knee_l_name, theta_deg))

    # Faz o estado atual virar o default do modelo
    model.setPropertiesFromState(state)

    # Salva novo modelo
    model.printToXML(output_model_path)

    print("\n=== MODELO GERADO ===")
    for nome, ang in alteradas:
        print(f"{nome} = {ang} graus")
    print(f"Arquivo salvo em: {output_model_path}")

    return model


def criar_modelo_com_flexao_bilateral(
    model_path,
    output_model_path,
    theta_esq_deg,
    phi_esq_deg,
    alpha_esq_deg,
    theta_dir_deg,
    phi_dir_deg,
    alpha_dir_deg
):
    """
    Cria um modelo com flexões diferentes para o lado esquerdo e direito.
    
    Parâmetros:
    - theta_esq_deg, phi_esq_deg, alpha_esq_deg: ângulos do lado esquerdo (perna, coxa, pé)
    - theta_dir_deg, phi_dir_deg, alpha_dir_deg: ângulos do lado direito (perna, coxa, pé)
    """
    model = osim.Model(model_path)
    state = model.initSystem()

    coord_set = model.getCoordinateSet()

    # Converter graus para radianos
    theta_esq_rad = np.deg2rad(theta_esq_deg)
    phi_esq_rad = np.deg2rad(phi_esq_deg)
    alpha_esq_rad = np.deg2rad(alpha_esq_deg)
    
    theta_dir_rad = np.deg2rad(theta_dir_deg)
    phi_dir_rad = np.deg2rad(phi_dir_deg)
    alpha_dir_rad = np.deg2rad(alpha_dir_deg)

    # Nomes das coordenadas
    knee_r_name = "knee_angle_r"
    knee_l_name = "knee_angle_l"
    hip_r_name = "hip_flexion_r"
    hip_l_name = "hip_flexion_l"
    ankle_r_name = "ankle_angle_r"
    ankle_l_name = "ankle_angle_l"

    alteradas = []

    # --- LADO DIREITO ---
    if coord_set.contains(knee_r_name):
        coord_set.get(knee_r_name).setValue(state, theta_dir_rad, True)
        alteradas.append((knee_r_name, theta_dir_deg))
    if coord_set.contains(hip_r_name):
        coord_set.get(hip_r_name).setValue(state, phi_dir_rad, True)
        alteradas.append((hip_r_name, phi_dir_deg))
    if coord_set.contains(ankle_r_name):
        coord_set.get(ankle_r_name).setValue(state, alpha_dir_rad, True)
        alteradas.append((ankle_r_name, alpha_dir_deg))

    # --- LADO ESQUERDO ---
    if coord_set.contains(knee_l_name):
        coord_set.get(knee_l_name).setValue(state, theta_esq_rad, True)
        alteradas.append((knee_l_name, theta_esq_deg))
    if coord_set.contains(hip_l_name):
        coord_set.get(hip_l_name).setValue(state, phi_esq_rad, True)
        alteradas.append((hip_l_name, phi_esq_deg))
    if coord_set.contains(ankle_l_name):
        coord_set.get(ankle_l_name).setValue(state, alpha_esq_rad, True)
        alteradas.append((ankle_l_name, alpha_esq_deg))

    # Faz o estado atual virar o default do modelo
    model.setPropertiesFromState(state)

    # Salva novo modelo
    model.printToXML(output_model_path)

    print("\n=== MODELO BILATERAL GERADO ===")
    print("--- LADO DIREITO ---")
    for nome, ang in alteradas:
        if "_r" in nome:
            print(f"{nome} = {ang} graus")
    print("--- LADO ESQUERDO ---")
    for nome, ang in alteradas:
        if "_l" in nome:
            print(f"{nome} = {ang} graus")
    print(f"Arquivo salvo em: {output_model_path}")

    return model


if __name__ == "__main__":
    model_path = r"./Rajagopal_2015.osim"
    output_model_path = r"./calibrated_Rajagopal_2015_joelho_flexionado-3.osim"

    # 1) listar todas as coordenadas
    listar_coordenadas(model_path)

    # 2) identificar joelhos
    identificar_joelhos(model_path)

    # 3) criar uma nova versão com theta graus de flexão
    criar_modelo_com_flexao_joelhos(
        model_path=model_path,
        output_model_path=output_model_path,
        theta_deg=45,         # exemplo
        aplicar_em="esquerdo"    # "ambos", "direito" ou "esquerdo"
    )