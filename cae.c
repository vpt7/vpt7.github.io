#include <stdio.h>

int main() {
    char msg[] = "hello how are u";
    int key = 3;  // shift value
    char ch;
    printf("Original: %s\n", msg);

    // Encrypt
    for (int i = 0; msg[i] != '\0'; i++) {
        ch = msg[i];
        if (ch >= 'a' && ch <= 'z')
            msg[i] = (ch - 'a' + key) % 26 + 'a';
    }
    printf("Encrypted: %s\n", msg);

    // Decrypt
    for (int i = 0; msg[i] != '\0'; i++) {
        ch = msg[i];
        if (ch >= 'a' && ch <= 'z')
            msg[i] = (ch - 'a' - key + 26) % 26 + 'a';
    }
    printf("Decrypted: %s\n", msg);

    return 0;
}
