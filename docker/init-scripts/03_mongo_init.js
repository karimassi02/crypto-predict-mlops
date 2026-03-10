// ============================================
// Phase 4: Securite - Utilisateur MongoDB
// ============================================

db = db.getSiblingDB("crypto_predict");

db.createUser({
  user: "crypto_app",
  pwd: "crypto_app_pwd",
  roles: [
    { role: "readWrite", db: "crypto_predict" }
  ]
});

db.createUser({
  user: "crypto_readonly",
  pwd: "crypto_readonly_pwd",
  roles: [
    { role: "read", db: "crypto_predict" }
  ]
});

print("MongoDB users created for crypto_predict database");
