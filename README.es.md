<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/prism-verify/main/assets/prism-verify-logo.png" alt="prism-verify logo" width="500">
</p>

<p align="center">
  <a href="https://pypi.org/project/prism-verify/"><img src="https://img.shields.io/pypi/v/prism-verify" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/@mcptoolshop/prism-verify"><img src="https://img.shields.io/npm/v/@mcptoolshop/prism-verify" alt="npm"></a>
  <a href="https://mcp-tool-shop-org.github.io/prism-verify/"><img src="https://img.shields.io/badge/Landing_Page-live-22d3ee" alt="Landing Page"></a>
  <a href="https://mcp-tool-shop-org.github.io/prism-verify/handbook/"><img src="https://img.shields.io/badge/Handbook-docs-22d3ee" alt="Handbook"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
</p>

# 

Servicio de evaluación en tiempo de ejecución para flujos de trabajo de agentes. Verificación multifacética, sin razonamiento y con diferentes familias, con registros reproducibles: para código, llamadas a herramientas, citas y la **adulación** en las respuestas. **[Página de inicio y manual →](https://mcp-tool-shop-org.github.io/prism-verify/)**

## Instalar

Instale la CLI `prism` (y el servicio HTTP) en su PATH:

```bash
uv tool install prism-verify        # or: pipx install prism-verify
```

¿No usa Python? Use el iniciador de npm (descarga y verifica SHA256 de un binario precompilado):

```bash
npx @mcptoolshop/prism-verify verify --artifact @file.py --intent "..." --caller-family openai
```

O agréguelo como una biblioteca; extras: `[anthropic]` `[openai]` `[google]` `[mcp]` `[http]` `[all]`:

```bash
uv add prism-verify
# or
pip install "prism-verify[all]"
```

## Inicio rápido

Prism siempre realiza la validación con una familia de modelos **diferente** a la del llamante (Bloqueo 1), por lo que
configure al menos un proveedor de familia alternativa. Genere una clave de firma Ed25519 (la predeterminada; los comprobantes pueden ser verificados por cualquier persona que tenga la clave pública) para que se puedan generar los comprobantes, o utilice
`PRISM_DEV=1` para pruebas locales:

```bash
prism keygen --out ~/.prism/signing_key.pem      # Ed25519 keypair (default signing)
export PRISM_SIGNING_KEY=~/.prism/signing_key.pem # or: export PRISM_DEV=1 (local play)
export ANTHROPIC_API_KEY="sk-ant-..."             # alt-family verifier for an OpenAI-family caller

prism verify \
  --artifact @myfile.py \
  --intent "Sort a list in O(n log n)" \
  --caller-family openai \
  --provider anthropic
```

> Alternativa heredada: `export PRISM_SIGNING_SECRET="$(openssl rand -hex 32)"` firma los comprobantes con
> HMAC en su lugar (solo puede ser verificado por los titulares de ese secreto compartido; consulte [Comprobantes](#receipts--signing-ed25519-verifiable-by-anyone)).

## Arquitectura

Prism aplica cuatro bloqueos arquitectónicos en el contrato de la API:

1. **Familia diferente:** la familia de modelos del llamante siempre se excluye de la validación.
2. **Sin razonamiento:** se elimina el CoT del productor antes de cruzar el límite de la familia.
3. **Múltiples lentes:** se ejecutan al menos 3 lentes independientes en paralelo.
4. **Con conocimiento de la submodularidad:** se rechaza si los lentes están demasiado de acuerdo (señal colapsada).

Para los artefactos de **cita**, se aplica una capa preliminar antes de la verificación de la "fundamentación" del LLM; cada etapa determinista rechaza aquello que puede *demostrar*, y en caso contrario, se abstiene:

- **Capa de existencia:** recuperación en vivo de arXiv/Crossref; un identificador fabricado se descarta, sin analizarlo.
- **Capa numérica/de unidades:** se detecta aritméticamente un cambio porcentual, un error en la escala de unidades (42 mili- frente a micro-segundos de arco) o una falsedad en la dirección de la comparación (5.0 < 5.8 ≠ "superado").
- **Verificación de la fundamentación:** verificación del LLM, con diferentes familias y sin razonamiento, contra el resumen recuperado.
- **Capa NLI ortogonal** *(opcional, `PRISM_NLI_FLOOR`)*: un codificador NLI (reconocimiento de implicaciones naturales) cruza la información con un modelo diferente para vetar una respuesta "apoyada" que haya dado el LLM, pero que un modelo mecánicamente diferente no corrobore.

### Utilice su propio verificador

La verificación de la fundamentación puede ejecutarse contra un modelo **que usted aloje** en lugar de una API alojada; para ello, active la opción a través de `PRISM_LOCAL_VERIFIER_ENDPOINT`, que utiliza diferentes familias y permite fallar de forma segura hacia sus verificadores alojados. La comprobación más frecuente no tiene costo por llamada y su evidencia se mantiene local. Un receptor de captura opcional (`PRISM_HARVEST_PATH`) registra las tripletas `(afirmación, evidencia, veredicto)` para que pueda entrenar uno. Consulte el [manual](https://mcp-tool-shop-org.github.io/prism-verify/handbook/local-verifier/).

### Adulación (verificación de la respuesta)

Más allá del código, las llamadas a herramientas y las citas, Prism evalúa una **RESPUESTA** del modelo para detectar una **adulación** *regresiva*: decirle al usuario lo que quiere oír en lugar de lo que es correcto (afirmar una premisa falsa, abandonar una respuesta correcta ante una simple objeción). Ejecuta un especialista afinado con diferentes familias y sin razonamiento como la "lente" de adulación; para ello, active la opción a través de `PRISM_SYCOPHANCY_ENDPOINT`, que permite fallar de forma segura **absteniéndose** (nunca con un silencio que indique "no es adulador"). El acuerdo con un usuario *correcto* o la concesión ante una refutación bien fundamentada son señales de honestidad, no de adulación. Consulte el [manual](https://mcp-tool-shop-org.github.io/prism-verify/handbook/).

## Calibración y prueba de rendimiento (`prism eval`)

Prism está diseñado para ser **medido**, no solo para hacer afirmaciones. `prism eval` ejecuta los modelos sobre un corpus etiquetado y genera informes —basados en los propios datos de Prism— sobre la precisión/exhaustividad/coeficiente de correlación de Matthews (MCC) por modelo, la matriz de diversidad entre modelos (alfa de Krippendorff + kappa de Cohen por pares), la ganancia de cobertura submodular, la precisión de la decisión y la calibración de la confianza (ECE/Brier), todo ello con un intervalo de confianza honesto.

```bash
prism eval --split all --runs 1        # measure against the bundled corpus (needs a verifier)
prism eval --offline                    # deterministic mock (CI smoke; NOT a real measurement)
```

**Medido, no afirmado** ([última ejecución →](eval/RESULTS.md): Ollama local `mistral-small:24b`, 111 muestras, 2026-06-14):

| Lo que medimos | Resultado |
|---|---|
| MCC por lente (contrato / cruce de límites / invariante / fundamentación) | 0.33 / 0.71 / 0.48 / **0.78** |
| Independencia de las lentes (alfa de Krippendorff), pero contrato↔invariante es redundante (kappa de Cohen) | α 0.162 · **κ 0.717** |
| Precisión/calibración general del veredicto (ECE) | 0.667 / 0.241 |
| Comprobación honesta de la contaminación: precisión con datos contaminados frente a datos no contaminados | 0.560 frente a **0.754** (Δ 0.194) |
| Seguridad de las citas: una cita incorrecta nunca se acepta ciegamente (recuperación de rechazos) | **1.000** |

El corpus es pequeño y los principales hallazgos son *indicativos*, pero son mediciones reales, no simuladas, y el informe es honesto sobre lo que muestra y lo que no (por ejemplo, la ganancia en cobertura es 0 en este corpus con predominio de invariantes). Consulte el [manual de evaluación](https://mcp-tool-shop-org.github.io/prism-verify/handbook/evaluation/) para conocer el método y un ejemplo práctico.

## Servicio HTTP

Ejecute prism como un servicio HTTP (necesita el extra `[http]`):

```bash
prism serve --host 127.0.0.1 --port 8000      # OpenAPI docs at /docs
```

| Punto final | Qué hace |
|---|---|
| `POST /verify` | Valida un artefacto (el mismo contrato que la CLI). Se bloquea dentro del presupuesto; `Prefer: respond-async` + una URL de `webhook` → `202`, el resultado se entrega al webhook (firmado). |
| `GET /replay/{receipt_id}` | El comprobante firmado + `signature_valid`. |
| `POST /verify-receipt` | Valida un comprobante independiente (entre herramientas). |
| `GET /healthz` | Estado activo + familias de validadores configuradas (sin autenticación). |

Establezca las claves de la API (almacenadas de forma segura); prism es **fail-closed**, por lo que `/verify` se rechaza hasta que se configuren las claves
o se opte por la opción de no autenticación local:

```bash
export PRISM_API_KEYS="<sha256(key1)>,<sha256(key2)>"   # callers send: Authorization: Bearer <key>
export PRISM_WEBHOOK_SECRET="<random>"                  # to sign async/escalate webhook deliveries
# local dev only:
export PRISM_HTTP_ALLOW_NO_AUTH=1
```

Los errores son RFC 9457 `application/problem+json`; `POST /verify` respeta una cabecera `Idempotency-Key`
y un límite de velocidad por clave (`429` + `Retry-After`). Los webhooks asíncronos/de escalamiento son
Standard-Webhooks-signed, con protección SSRF (sin destinos internos/de metadatos), se reintentan y llevan un
compensador de eventos de cancelación con nombre.

## Comprobantes y firma (Ed25519, verificable por cualquier persona)

Cada validación produce un comprobante firmado y reproducible en `~/.prism/receipts.db`. v0.4 firma
los nuevos comprobantes con **Ed25519 (RFC 8032)** de forma predeterminada, por lo que **una herramienta diferente puede verificar un comprobante de prism con la clave pública de prism; no se necesita una clave compartida**:

```bash
prism keygen --out ~/.prism/signing_key.pem    # generate an Ed25519 keypair
export PRISM_SIGNING_KEY=~/.prism/signing_key.pem
prism pubkey                                    # publish this public key + kid to consumers

# a consumer (e.g. role-os) verifies a receipt with ONLY the public key:
prism verify-receipt receipt.json --public-key prism-pub.pem
```

La firma cubre el resultado, los hashes del artefacto pre/post-strip, el modelo de validador, la
matriz de submodularidad, los hashes de los prompts por lente (reproducibles byte por byte), los pines de recuperación de citas y el `alg`/`kid` de firma. Los comprobantes **HMAC** heredados siguen siendo válidos (establezca
`PRISM_SIGNING_SECRET`); `PRISM_DEV=1` genera una clave de desarrollo para pruebas locales. Prism **se niega a iniciar**
los caminos de verificación/reproducción/servicio/MCP si no se configura ninguna clave, en lugar de firmar silenciosamente
con una clave conocida públicamente.

Administre los comprobantes almacenados con los comandos del compensador:

```bash
prism receipt delete <receipt_id>
prism receipt prune --older-than 90d --yes
```

## Seguridad y privacidad

- **Modelo de amenazas.** Prism lee el artefacto + la intención que se le pasa y las respuestas de los modelos de validador, y escribe comprobantes firmados en una base de datos SQLite local. **No** lee
su árbol de código fuente, entorno ni credenciales más allá de las claves de la API del proveedor que proporciona a través de
las variables de entorno. Las firmas de los comprobantes brindan **verificabilidad de terceros** (Ed25519: un
consumidor verifica con la clave pública, sin clave compartida), pero no son a prueba de manipulaciones contra un
atacante con acceso raíz local que pueda leer la clave privada almacenada; este es el mismo límite que la clave secreta HMAC. Para una resistencia genuina a las manipulaciones, almacene la clave en un HSM y ancle los comprobantes en un
registro de transparencia (el camino de endurecimiento con nombre).
- **Superficie HTTP.** `prism serve` se enlaza a la interfaz de bucle local de forma predeterminada, es **fail-closed** (sin `/verify`
sin claves de la API), almacena las claves de forma segura y **protege contra SSRF** las URL de webhook proporcionadas por el llamante
(sin destinos internos/de enlace local/de metadatos). Ejecuta los artefactos proporcionados por el llamante a través de un modelo;
un artefacto puede *intentar* la inyección de prompts, pero no puede cambiar el esquema del resultado ni extraer
las claves del proveedor de prism.
- **Sin telemetría.** Prism envía solicitudes solo a los proveedores de modelos que configure
(Anthropic / OpenAI / Google / Ollama local). Nada más.
- Política completa: [SECURITY.md](SECURITY.md).

## Licencia

MIT
