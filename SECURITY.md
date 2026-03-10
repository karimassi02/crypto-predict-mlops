# Politique de Securisation des Donnees

> Document requis pour la competence C1.4.1 — RNCP39586

---

## 1. Gestion des secrets

### Principes
- Les secrets (cles API, mots de passe) sont stockes dans des **variables d'environnement** via un fichier `.env`.
- Le fichier `.env` est **exclu du controle de version** (`.gitignore`).
- Un fichier `.env.example` documente les variables requises sans exposer de valeurs reelles.
- La fonction `validate_env()` verifie au demarrage que tous les secrets obligatoires sont configures et alerte sur les mots de passe faibles.
- Les secrets sont masques dans les logs via `mask_secret()` (ex: `****xy2z`).

### Variables sensibles
| Variable | Usage | Obligatoire |
|----------|-------|:-----------:|
| `COINGECKO_API_KEY` | Acces a l'API CoinGecko | Oui |
| `POSTGRES_PASSWORD` | Authentification PostgreSQL | Oui |
| `MONGO_ROOT_PASSWORD` | Admin MongoDB | Non (dev) |
| `AIRFLOW_ADMIN_PASSWORD` | Interface Airflow | Non (dev) |

---

## 2. Securisation PostgreSQL

### Roles et privileges
Trois niveaux d'acces sont definis :

| Role | Droits | Usage |
|------|--------|-------|
| `admin` | Superuser | Administration uniquement |
| `app_readwrite` | SELECT, INSERT, UPDATE | Collecteurs, pipeline ETL |
| `app_readonly` | SELECT | Dashboards, analyses |

### Mesures appliquees
- **Principe du moindre privilege** : chaque composant utilise le role minimum necessaire.
- **Pas de DELETE accorde** au role applicatif — les donnees sont append-only.
- Les privileges publics par defaut sont revoques (`REVOKE ALL ON DATABASE crypto_predict FROM PUBLIC`).
- Les scripts d'initialisation (`02_create_roles.sql`) sont executes automatiquement au premier demarrage.

---

## 3. Securisation MongoDB

### Authentification
- L'authentification est activee via `MONGO_INITDB_ROOT_USERNAME` / `MONGO_INITDB_ROOT_PASSWORD`.
- Deux utilisateurs applicatifs sont crees automatiquement :
  - `crypto_app` (readWrite sur `crypto_predict`) — pour l'insertion des news.
  - `crypto_readonly` (read sur `crypto_predict`) — pour la consultation.
- Les URI de connexion incluent les credentials (`mongodb://user:pwd@host:port/db`).

---

## 4. Securisation reseau (Docker)

### Architecture
- Tous les services communiquent via un **reseau Docker interne** (`crypto_network`).
- Les ports sont exposes sur `localhost` uniquement (pas de `0.0.0.0` en production).
- En production, les ports des bases de donnees (`5432`, `27017`) doivent etre **fermes** au trafic externe.

### Recommandations production
- Utiliser un reverse proxy (nginx/traefik) devant Airflow.
- Activer SSL/TLS pour PostgreSQL (`sslmode=require`).
- Configurer un pare-feu pour restreindre les acces reseau.

---

## 5. Validation et sanitization des donnees

### Donnees d'entree
- **API responses** : validation des types (numeriques, dates) avant insertion en base.
- **Web scraping** : sanitization du texte (`sanitize_text()`) pour supprimer les caracteres de controle et echapper le HTML.
- **URLs** : validation du schema (`sanitize_url()`) — seuls `http://` et `https://` sont acceptes.
- **Identifiants crypto** : validation par regex (`validate_crypto_id()`) — alphanumeriques et tirets uniquement.

### Protection contre les injections
- **SQL** : utilisation de requetes parametrees (`%s`) via psycopg2 — aucune concatenation de chaines.
- **NoSQL** : utilisation de l'API pymongo avec des dictionnaires Python — pas de construction de requetes par chaines.
- **XSS** : echappement HTML des donnees scrapees avant stockage.

---

## 6. Bonnes pratiques appliquees

| Pratique | Implementation |
|----------|---------------|
| Secrets hors du code | `.env` + `.gitignore` |
| Validation au demarrage | `validate_env()` |
| Moindre privilege | Roles PostgreSQL / MongoDB |
| Requetes parametrees | psycopg2 `execute_values` |
| Sanitization des inputs | `security.py` |
| Masquage des secrets dans les logs | `mask_secret()` |
| Reseau isole | Docker bridge network |
| Healthchecks | Sur tous les services Docker |

---

## 7. Axes d'amelioration (production)

- Utiliser un gestionnaire de secrets (HashiCorp Vault, AWS Secrets Manager).
- Activer le chiffrement au repos pour PostgreSQL et MongoDB.
- Mettre en place une rotation automatique des mots de passe.
- Ajouter du rate limiting sur l'API FastAPI (Phase 6).
- Configurer l'audit logging sur les bases de donnees.
- Implementer HTTPS avec certificats Let's Encrypt.
