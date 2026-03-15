/*
 * TweetNaCl — Ed25519 subset for NML program signing
 * Public domain. Original: https://tweetnacl.cr.yp.to/
 */
#ifndef TWEETNACL_H
#define TWEETNACL_H

#define crypto_sign_BYTES 64
#define crypto_sign_PUBLICKEYBYTES 32
#define crypto_sign_SECRETKEYBYTES 64

extern int crypto_sign_keypair(unsigned char *pk, unsigned char *sk);
extern int crypto_sign(unsigned char *sm, unsigned long long *smlen,
                       const unsigned char *m, unsigned long long n,
                       const unsigned char *sk);
extern int crypto_sign_open(unsigned char *m, unsigned long long *mlen,
                            const unsigned char *sm, unsigned long long n,
                            const unsigned char *pk);
extern int crypto_hash(unsigned char *out, const unsigned char *m,
                       unsigned long long n);
extern int crypto_verify_32(const unsigned char *x, const unsigned char *y);
extern void randombytes(unsigned char *buf, unsigned long long len);

#endif
