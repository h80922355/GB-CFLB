"""
真实的密码学签名实现
基于ECDSA算法，符合区块链标准
"""
import hashlib
import json
from ecdsa import SigningKey, VerifyingKey, SECP256k1
from ecdsa.util import sigdecode_der, sigencode_der
import binascii


class CryptoSigner:
    """密码学签名器"""

    @staticmethod
    def generate_keypair():
        """
        生成ECDSA密钥对
        返回:
            (private_key_hex, public_key_hex)
        """
        # 生成私钥
        private_key = SigningKey.generate(curve=SECP256k1)
        public_key = private_key.get_verifying_key()

        # 转换为十六进制字符串
        private_key_hex = binascii.hexlify(private_key.to_string()).decode()
        public_key_hex = binascii.hexlify(public_key.to_string()).decode()

        return private_key_hex, public_key_hex

    @staticmethod
    def sign(message: str, private_key_hex: str) -> str:
        """
        使用私钥签名消息
        参数:
            message: 要签名的消息
            private_key_hex: 私钥的十六进制字符串
        返回:
            签名的十六进制字符串
        """
        # 恢复私钥对象
        private_key_bytes = binascii.unhexlify(private_key_hex)
        private_key = SigningKey.from_string(private_key_bytes, curve=SECP256k1)

        # 对消息进行哈希
        message_hash = hashlib.sha256(message.encode()).digest()

        # 签名
        signature = private_key.sign(message_hash, sigencode=sigencode_der)

        # 返回十六进制字符串
        return binascii.hexlify(signature).decode()

    @staticmethod
    def verify(message: str, signature_hex: str, public_key_hex: str) -> bool:
        """
        验证签名
        参数:
            message: 原始消息
            signature_hex: 签名的十六进制字符串
            public_key_hex: 公钥的十六进制字符串
        返回:
            验证结果
        """
        try:
            # 恢复公钥对象
            public_key_bytes = binascii.unhexlify(public_key_hex)
            public_key = VerifyingKey.from_string(public_key_bytes, curve=SECP256k1)

            # 恢复签名
            signature = binascii.unhexlify(signature_hex)

            # 对消息进行哈希
            message_hash = hashlib.sha256(message.encode()).digest()

            # 验证签名
            public_key.verify(signature, message_hash, sigdecode=sigdecode_der)
            return True
        except Exception:
            return False

    @staticmethod
    def compute_block_hash(block_header: dict) -> str:
        """
        计算区块哈希
        注意：区块哈希计算不包含QC，即使QC在header中

        参数:
            block_header: 区块头
        返回:
            区块哈希
        """
        # 按照设计文档的顺序构建哈希内容（不包含QC）
        hash_content = {
            'proposer_id': block_header.get('proposer_id'),
            'view': block_header.get('view'),
            'parent_hash': block_header.get('parent_hash'),
            'parent_qc': block_header.get('parent_qc'),
            'timestamp': block_header.get('timestamp'),
            'merkle_cluster': block_header.get('merkle_cluster'),
            'merkle_trans': block_header.get('merkle_trans')
        }
        # 注意：即使block_header中有'qc'字段，也不包含在哈希计算中

        # 序列化并计算哈希
        content_str = json.dumps(hash_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()