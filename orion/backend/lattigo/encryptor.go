package main

import (
	"C"

	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

//export NewEncryptor
func NewEncryptor() {
	if scheme.PublicKey == nil {
		return // no public key loaded; encryption not available (e.g. decrypt-only client)
	}
	scheme.Encryptor = ckks.NewEncryptor(*scheme.Params, scheme.PublicKey)
}

//export NewDecryptor
func NewDecryptor() {
	if scheme.SecretKey == nil {
		return // server side: no secret key loaded, decryption not available
	}
	scheme.Decryptor = ckks.NewDecryptor(*scheme.Params, scheme.SecretKey)
}

//export Encrypt
func Encrypt(plaintextID C.int) C.int {
	plaintext := RetrievePlaintext(int(plaintextID))
	ciphertext := ckks.NewCiphertext(*scheme.Params, 1, plaintext.Level())
	scheme.Encryptor.Encrypt(plaintext, ciphertext)

	idx := PushCiphertext(ciphertext)
	return C.int(idx)
}

//export Decrypt
func Decrypt(ciphertextID C.int) C.int {
	ciphertext := RetrieveCiphertext(int(ciphertextID))

	plaintext := ckks.NewPlaintext(*scheme.Params, ciphertext.Level())
	scheme.Decryptor.Decrypt(ciphertext, plaintext)

	idx := PushPlaintext(plaintext)
	return C.int(idx)
}
