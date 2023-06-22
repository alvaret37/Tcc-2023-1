import cv2
import face_recognition

# Carregar as imagens de referência e seus respectivos nomes
imagem_pessoa1 = face_recognition.load_image_file("alvaro.jpg")
imagem_pessoa2 = face_recognition.load_image_file("prof.felipe.jpg")
imagem_pessoa3 = face_recognition.load_image_file("Prof. Cleiton.jpg")
imagem_pessoa4 = face_recognition.load_image_file("Prof. Manoel.jpg")
imagem_pessoa5 = face_recognition.load_image_file("alexandre.jpg")

# Extrair os descritores faciais das imagens de referência
descritores_pessoa1 = face_recognition.face_encodings(imagem_pessoa1)[0]
descritores_pessoa2 = face_recognition.face_encodings(imagem_pessoa2)[0]
descritores_pessoa3 = face_recognition.face_encodings(imagem_pessoa3)[0]
descritores_pessoa4 = face_recognition.face_encodings(imagem_pessoa4)[0]
descritores_pessoa5 = face_recognition.face_encodings(imagem_pessoa5)[0]

# Lista dos descritores faciais conhecidos e seus respectivos nomes
descritores_conhecidos = [descritores_pessoa1, descritores_pessoa2, descritores_pessoa3, descritores_pessoa4, descritores_pessoa5]
nomes_conhecidos = ["Alvaro Emmanoel", "Prof.Felipe", "Prof. Cleiton", "Prof. Manoel", "Alexandre Filho"]

# Inicializar a captura de vídeo da webcam
captura = cv2.VideoCapture(0)

while True:
    # Capturar um quadro da webcam
    ret, frame = captura.read()

    # Redimensionar o quadro para melhor desempenho
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar as localizações das faces no quadro
    localizacoes = face_recognition.face_locations(frame_rgb)

    # Extrair os descritores faciais das faces detectadas
    descritores = face_recognition.face_encodings(frame_rgb, localizacoes)

    # Iterar sobre as faces detectadas
    for (top, right, bottom, left), descritor in zip(localizacoes, descritores):
        # Comparar os descritores faciais com os descritores conhecidos
        combinacoes = face_recognition.compare_faces(descritores_conhecidos, descritor)
        nome_pessoa = "Desconhecido"

        # Encontrar o nome correspondente ao descritor facial
        if True in combinacoes:
            indice = combinacoes.index(True)
            nome_pessoa = nomes_conhecidos[indice]

        # Desenhar um retângulo e exibir o nome da pessoa
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, nome_pessoa, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Exibir o quadro resultante
    cv2.imshow('Reconhecimento Facial', frame)

    # Parar o loop ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#libera a captura
captura.release()
cv2.destroyAllWindows()
